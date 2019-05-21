import sys,re,argparse
import numpy as np
from operator import itemgetter
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import scipy.sparse
import scipy
from numpy import linalg as LA
from scipy.sparse.linalg import norm as spnorm
def read_sim_matrix( infile, sparse ):
	if sparse:
		lineList = file(infile).read().split('\n')
		row, col, num = map(int, lineList[0].split())
		rowvec = np.zeros(num)
		colvec = np.zeros(num)
		data = np.zeros(num)
		for i in range(0, num):
			rowvec[i], colvec[i], data[i] = map(float, lineList[i + 1].split())
		outMtx = csr_matrix((data, (rowvec, colvec)), shape=(row, col))
		#return np.array(outMtx.todense())
		return outMtx.tolil()
	else:
		weightMtx = []
		for line in file(infile).readlines():
			weightMtx.append(np.array(map(float, line.split())))
		return np.array(weightMtx)

def remove_diag( inMtx ):
        
	num_row, num_col = np.shape(inMtx)
        print "num_row numcol " +str(num_row)+" "+str(num_col)
	for i in range(0, num_row):
		inMtx[i,i] = 0
	return inMtx

def check_valid( inMtx,sparse ):
	# check whether there are negative weights
        if sparse==1:
            if np.sum((inMtx.data<0).astype(np.int)) > 0:
                return 1
        else:
	    if np.sum((inMtx<0).astype(np.int)) > 0:
		return 1
	return 0

def keep_top( inMtx, K ):
	row, col = np.shape(inMtx)
        print "keep_top row is "+str(row)+" col is "+str(col)
	#outMtx = np.zeros((row, col))
	outMtx =  lil_matrix((row, col))
	for i in range(0, row):
		sortList = []
		for j in range(0, col):
			sortList.append((j, inMtx[i][j]))
		sortList = sorted(sortList, key=itemgetter(1), reverse=True)
		for k in range(0, K):
			j, s = sortList[k]
			outMtx[i,j] = s
	return outMtx.tocsr()

def check_size( Mtx1, Mtx2, Mtx12 ):
	row1, col1 = np.shape(Mtx1)
	row2, col2 = np.shape(Mtx2)
	row12, col12 = np.shape(Mtx12)
	if row1 != col1 or row2 != col2 or row1 != row12 or row2 != col12:
		return 1
	sys.stderr.write("#node in 1st layer: %d\n#node in 2nd layer: %d\n" %(row1, row2))
	return 0

def row_normalize( inMtx ):
        #print inMtx.dtype
        #data=inMtx.tocoo().data
        #col=inMtx.tocoo().col
        #rol=inMtx.tocoo().row
        row,col = np.shape(inMtx)
	sumd = np.squeeze(np.sum(inMtx, 1).getA())
        print sumd.shape
	#return (inMtx/sum[:, np.newaxis]).T
        return np.dot(scipy.sparse.spdiags(1./(sumd+1e-5),0,row,row),inMtx).tocsr()

def output_file( outfile, score ):
	fout = open(outfile, "w")
	for i in range(0, len(score)):
		#fout.write("%f\n" %score[i])
		fout.write(str(score[i,0])+"\n")
	fout.close()
	return

parser = argparse.ArgumentParser(description = "Running a two-layer MRRW algorithm.")
parser.add_argument('interweight1', metavar='layer1-edge-weight', help=': edge weight matrix for the 1st layer')
parser.add_argument('interweight2', metavar='layer2-edge-weight', help=': edge weight matrix for the 2nd layer')
parser.add_argument('betweenweight', metavar='layer1to2-edge-weight', help=': edge weight matrix between two layers')
parser.add_argument('--initscore1', metavar='layer1-init-score', help=': initial scores of nodes in the 1st layer (default = 1/N)')
parser.add_argument('--initscore2', metavar='layer2-init-score', help=': initial scores of nodes in the 2nd layer (default = 1/N)')
parser.add_argument('--outfile1', metavar='layer1-fin-score', help=': final scores of nodes in the 1st layer')
parser.add_argument('--outfile2', metavar='layer2-fin-score', help=': final scores of nodes in the 2nd layer')
parser.add_argument('-w', metavar='alpha', type=float, help=': weight for propagation (default = 0.9)', default=0.9)
parser.add_argument('-n', metavar='N', type=int, help=': leave top N highest within-layer edges for each nodes (default = all)')
parser.add_argument('-m', metavar='M', type=int, help=': leave top M highest between-layer edges for each nodes (default = all)')
parser.add_argument('-s', metavar='sparse', type=int, help='= 1 if the input matrices are sparse format (default = 0)', default=0)

args = parser.parse_args()
tmp=csr_matrix((3, 4), dtype=float)
print tmp
tmp[1,1]=0.5
tmp[0,0]=1
sumd=np.squeeze(np.sum(tmp,1).getA())
print sumd
print np.diag(1./(sumd+1e-5))

print (tmp.data<0)
print (tmp.data<0).dtype
sparse=0
if args.s == 1:
	sys.stderr.write("The input matrices are sparse format.\n")
	E_11 = remove_diag(read_sim_matrix(args.interweight1, 1).tolil())
	E_22 = remove_diag(read_sim_matrix(args.interweight2, 1).tolil())
	E_12 = read_sim_matrix(args.betweenweight, 1)
	E_21 = E_12.T
        sparse=1
else:
	E_11 = remove_diag(read_sim_matrix(args.interweight1, 0))
	E_22 = remove_diag(read_sim_matrix(args.interweight2, 0))
	E_12 = read_sim_matrix(args.betweenweight, 0)
	E_21 = E_12.T
        sparse=0
print "sparse = "+str(sparse)
if check_size(E_11, E_22, E_12):
	sys.stderr.write("Error: The dimensions do not match.\n")
	exit()

if args.n != None:
	E_11 = keep_top(E_11, args.n)
	E_22 = keep_top(E_22, args.n)

if args.m != None:
	E_21 = keep_top(E_12.T, args.m)
	E_12 = keep_top(E_12, args.m)

#if check_valid(E_11,args.s) or check_valid(E_22,args.s) or check_valid(E_12,args.s):
#	sys.stderr.write("Error: The input weights have negative values.\n")
#	exit()

num1 = np.shape(E_11)[0]
num2 = np.shape(E_22)[0]

L_11 = row_normalize(E_11)
print "finish normalize L_11"
L_22 = row_normalize(E_22)
L_12 = row_normalize(E_21)
L_21 = row_normalize(E_12)
print "finish normalize L_12"
print np.shape(L_12)

# initial score setting
#S1 = np.ones(num1)/num1
#if args.initscore1 != None:
#	S1 = read_sim_matrix(args.interweight1)
#	if check_valid(S1,sparse):
#		sys.stderr.write("Error: Some predefined scores are negative.\n")
#		exit()
#S1 = S1[:, np.newaxis]
		
#S2 = np.ones(num2)/num2
#if args.initscore1 != None:
#	S2 = read_sim_matrix(args.interweight1)
#	if check_valid(S2,sparse):
#		sys.stderr.write("Error: Some predefined scores are negative.\n")
#		exit()
#S2 = S2[:, np.newaxis]

#former = (1 - args.w) * S1 + args.w * (1 - args.w) * np.dot(L_11, np.dot(L_12, S2))
#latter = args.w * args.w * np.dot(L_11, np.dot(L_12, np.dot(L_22, L_21)))
#combine = former * np.ones(num1) + latter

#w, v = LA.eig(combine)

#score1=csr_matrix(np.random.rand(num1,1))
#score2=csr_matrix(np.random.rand(num2,1))
#print np.ones((num1,1))/num1
score1=csr_matrix(np.ones((num1,1))/num1)
score2=csr_matrix(np.ones((num2,1))/num2)
for i in range(100):
    prevs1=score1
    prevs2=score2
    #print L_21.shape
    #print score2.shape 
    #tmp1=np.dot(L_21, score2)
    #print tmp1.shape
    score1 = (1 - args.w) * score1 + args.w * np.dot(L_11, np.dot(L_21, score2))
    score2 = (1 - args.w) * score2 + args.w * np.dot(L_22, np.dot(L_12, score1))
    #print "score1 shape"
    #print np.shape(score1)
    #print "prevs1 shape"
    #print np.shape(prevs1)
    #print prevs1-score1
    #difs1=np.sum(np.absolute(prevs1-score1),0)
    #difs1=np.sum(np.absolute(prevs1-score1),0)
    score1=score1/(np.sum(np.sum(score1.todense())))
    score2=score2/(np.sum(np.sum(score2.todense())))
    #difs1=LA.norm((prevs1-score1).todense().getA(),ord=2)
    #difs2=LA.norm((prevs2-score2).todense().getA(),ord=2)
    difs1=spnorm(prevs1-score1,'fro')
    difs2=spnorm(prevs2-score2,'fro')
    print str(i)+"th iter : dif1 is "+str(difs1)+" dif2 is "+str(difs2)
#score1 = abs(v[:, 0])/sum(abs(v[:, 0]))
#score2 = (1 - args.w) * S2 + args.w * np.dot(L_22, np.dot(L_21, S1))

if args.outfile1 != None:
	output_file(args.outfile1, score1.todense())
else:
	sys.stdout.write("Scores for layer1: ")
	for i in range(0, num1):
		sys.stdout.write("%f " %score1[i])
	sys.stdout.write("\n")

if args.outfile2 != None:
	output_file(args.outfile2, score2.todense())
else:
	sys.stdout.write("Scores for layer2: ")
	for i in range(0, num2):
		sys.stdout.write("%f " %score2[i])
	sys.stdout.write("\n")

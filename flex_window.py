#!/usr/bin/python
# -*- coding: UTF-8 -*-

import gzip  
import math
import string
import operator
import pickle
import csv
import json
import uuid
import random
import subprocess
import sys, re, os
import multiprocessing
import numpy as np
from scipy.special import comb,perm
from collections import OrderedDict
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
from collections import defaultdict
from random import shuffle
from nltk.corpus import stopwords

def load_qrets(target):
	Dict ={}
	count = 0
	with open(target) as fin:
		for line in fin.readlines():
			qid,_,docno,rank,score,method = line.strip().split(' ')
			if qid not in Dict:
				Dict[qid] ={'docno':[],'rank':[],'bm25':[]}
			Dict[qid]['docno'].append(docno)
			Dict[qid]['rank'].append(int(rank))
			Dict[qid]['bm25'].append(float(score))
			count+=1
	print 'processing %s: %d queries %d documents'%(target,len(Dict),count)
	return Dict

def load_qrels(file):
	qrel_data ={}
	with open (file) as fin:
		for line in fin.readlines():#401 0 WT01-B04-284 0
			qid,_,docno,tag =line.strip().split(' ')
			if tag !='0':
				if qid not in qrel_data:
					qrel_data[qid] ={'docno':[]}
				qrel_data[qid]['docno'].append(docno)
	return qrel_data

def term2vector(terms):
	vectors =[]
	for term in terms:
		try:
			vectors.append(w2v[term])
		except:
			vectors.append(np.random.uniform(-0.25,0.25,Embedding))
	vectors = np.array(vectors)
	return vectors

def rolling_max(A, window, num_max):
	'''Computes roling maximum of 2D array.
	A is the array, window is the length of the rolling window and num_max is the number of maximums to return for each window.
	The output is an array of size (D,N,num_max) where D is the number of 
	columns in A and N is the number of rows.
	'''
	shape = (A.shape[1], np.max([A.shape[0]-window+1, 1]), np.min([window, A.shape[0]]))
	strides = (A.strides[-1],) + (A.strides[-1]*A.shape[1],) + (A.strides[-1]*A.shape[1],)
	b = np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)
	return np.sort(b, axis=2)[:,:,::-1][:,:,:num_max]

def _weighting_top2(query_np, doc_np, weights, wind):
	Doc_Score =[]
	try:
		query_np = query_tv(query_np)#term vectors weighting
		temp = doc_np.dot(query_np.T)/np.outer(np.linalg.norm(doc_np, axis=1),np.linalg.norm(query_np, axis=1))#doc_len*query_len
		if wind > doc_np.shape[0]:
			length = doc_np.shape[0]
		else:
			length = wind
		TopK = int(round(math.log(length)))+1
		Con_Maxs = rolling_max(temp, length, TopK)#[query_len*(doc_len-45+1)*2] 
		
		Con_Score = np.sum(Con_Maxs, axis=0)#[(doc_len-45+1)*2]
		score_al = []

		'''for alpha in weights:#Max1+alpha*Max2
			al = np.max(np.array(Con_Score)[:,0] + alpha*np.array(Con_Score)[:,1])
			score_al.append(al.tolist())
			# score_al.append(al.tolist()*math.log(co+45))#'''

		for alpha in weights:#Max1+alpha*meanwid/logwid
			al = np.max(np.array(Con_Score)[:,0] + alpha*np.array(np.mean(np.array(Con_Score),axis=1)))#Max1+ meanwid
			# al = np.max(np.array(Con_Score)[:,0] + alpha*np.log(np.abs(np.array(np.mean(np.array(Con_Score),axis=1)))))#Max1+ logwid
			score_al.append(al.tolist()*math.log(co+15))#''' 15 is constant C

		Doc_Score.append(score_al)
	except:
		Doc_Score.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

	return Doc_Score

def query_tv(query_np):##2 通过自身向量的计算
	gi = np.multiply(np.eye(query_np.shape[0],dtype=float),np.dot(query_np,query_np.T))
	g = [math.exp(item) for item in np.sum(gi,axis=1)]
	g = [[item/np.sum(g)] for item in g]

	g = np.repeat(np.array(g),query_np.shape[1],axis=1)
	query_np = np.multiply(query_np,g)
	return query_np

def _term2vector(query, doc):
	vectors =[]
	terms = list(set(query)|set(doc))
	for term in terms:
		try:
			vectors.append(w2v[term])
		except:
			vectors.append(np.random.uniform(-0.25, 0.25, Embedding))

	query_vectors =[]
	for word in query:
		query_vectors.append(vectors[terms.index(word)])
	doc_vectors =[]
	for word in doc:
		doc_vectors.append(vectors[terms.index(word)])
	return np.array(query_vectors),np.array(doc_vectors)

def window_top2_method(qid, docno, wind, weights):
	qry_list = pickle.load(open(os.path.join(query_path, qid)))
	sents = pickle.load(open(os.path.join(doc_path, docno)))
	words = sum(sents, [])

	qry_np = term2vector(qry_list)
	doc_np = term2vector(words)
	# qry_np,doc_np = _term2vector(qry_list,words)

	temp = qry_np.dot(qry_np.T)/np.outer(np.linalg.norm(qry_np, axis=1),np.linalg.norm(qry_np, axis=1))#query_len*query_len
	# cosin_dis = (len(qry_list)+np.sum(temp))/2

	cos_dis = temp#[temp<0.99999]
	if len(qry_list) >= 2:
		wind = (Cons_1+1)*int(round(len(qry_list)*np.exp((-1)*np.square(np.mean(cos_dis))/(2*np.var(cos_dis)))))+Cons_2#  qrylen*exp(-mean^2/(2*var))//gaussian
	else:
		wind = Cons_2
		# wind =len(qry_list)*15#'''

	# wind = (Cons_1+1)*len(qry_list)+Cons_2 #linear function of querylen

	global co 
	co = len(set(qry_list)&set(words))#统计查询里有多少个单词出现在文档里
	# co = sum([words.count(term) for term in qry_list])/float(len(qry_list))

	doc_scores = _weighting_top2(qry_np, doc_np, weights, wind)
	return doc_scores

def qid_docs_scores(target,wind,weights):
	if os.path.isfile(target) and target.find('.res')!=-1:#是bm25的qret文件
		prediction_data_bm25 = load_qrets(target)
		prediction_data ={}
		count=0
		for qid,value in prediction_data_bm25.items():
			count+=1
			print 'processing qid %s: %d / %d'%(qid,count,len(prediction_data_bm25))
			y_pred =[]
			for docno in value['docno']:
				doc_scores = window_top2_method(qid, docno, wind, weights)# windows+top2+weights
				
				y_pred.append(doc_scores)
			if qid not in prediction_data:
				prediction_data[qid] = {'docno':[],'y_pred':[]}
			prediction_data[qid]['docno']=value['docno']
			prediction_data[qid]['y_pred'].append(y_pred)
		return prediction_data
	
	elif os.path.isdir(target):#是docs的路径
		prediction_data ={}#存放qid，docno 和分值
		count=0
		for qid in os.listdir(query_path):
			count+=1
			print 'processing qid %s: %d / %d'%(qid,count,len(os.listdir(query_path)))
			y_pred =[]
			y_doc =[]
			for docno in os.listdir(target):
				doc_scores = score(qid, docno, wind, weights)# windows+top2+weights
				y_pred.append(doc_scores)
				y_doc.append(docno)
			if qid not in prediction_data:
				prediction_data[qid] = {'docno':[],'y_pred':[]}
			prediction_data[qid]['docno'].append(y_doc)
			prediction_data[qid]['y_pred'].append(y_pred)
		return prediction_data

def Rank_topK(qid_docno_score_dict, idx, topk=1000):
	res_dict = {'questions': []}
	for qid ,value in qid_docno_score_dict.items():
		docnos = value['docno']
		scores = value['y_pred']
		scores = np.array(scores[0])
		scores = scores[:,0,idx]
		retr_scores = list(zip(docnos, scores))
		shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
		sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
		# try:
			# sorted_retr_scores = sorted_retr_scores[:topk]
		# except:
			# sorted_retr_scores = sorted_retr_scores
		res_dict['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
	return res_dict

def Combine5_bm25(qid_docno_score_dict, prediction_data_bm25, idx, bm25weighting, topk=1000):
	res_dict = {'questions': []}
	for qid ,value in qid_docno_score_dict.items():
		docnos = value['docno']
		scores = value['y_pred']
		scores = np.array(scores[0])
		scores = scores[:,0,idx]
		bm25_scores = prediction_data_bm25[qid]['bm25']
		scores = (bm25weighting*np.array(bm25_scores)+scores).tolist()
		retr_scores = list(zip(docnos, scores))
		shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
		sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
		# try:
			# sorted_retr_scores = sorted_retr_scores[:topk]
		# except:
			# sorted_retr_scores = sorted_retr_scores
		res_dict['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
	return res_dict

def Combine_bm25(qid_docno_score_dict, prediction_data_bm25, idx, bm25weighting, idx_2, topk=1000):
	res_dict = {'questions': []}
	for qid ,value in qid_docno_score_dict.items():
		docnos = value['docno']
		scores = value['y_pred']#1*docnos*1*weights*weights
		scores = np.array(scores[0])
		scores = scores[:,0,idx_2,idx]
		bm25_scores = prediction_data_bm25[qid]['bm25']
		scores = (bm25weighting*np.array(bm25_scores)+(1.0-bm25weighting)*scores).tolist()
		retr_scores = list(zip(docnos, scores))
		shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
		sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
		res_dict['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
	return res_dict

def format_bioasq2treceval_qrels(qrel_data, filename):
	with open(filename, 'w') as f:
		for qid,value in qrel_data.items():
			for docno in value['docno']:
				f.write('{0} 0 {1} 1'.format(qid, docno))
				f.write('\n')

def format_bioasq2treceval_qret(qret_data, filename):
	with open(filename, 'w') as f:
		for q in qret_data['questions']:
			rank = 0a
			for d in q['documents']:
				# sim = (len(q['documents']) + 1 - rank) / float(len(q['documents']))
				sim =  q['score'][rank]
				f.write('{0} {1} {2} {3} {4} {5}'.format(q['id'], 0, d, rank, sim,'window_top2-bm25' ))
				f.write('\n')
				rank += 1

def trec_evaluate(qrels_file, qret_file, eval_name):
	trec_eval_res = subprocess.Popen(#os.path.dirname(os.path.realpath(__file__)) + '/./trec_eval
		['./eval/trec_eval', '-m', 'all_trec', qrels_file, qret_file],
		stdout=subprocess.PIPE, shell=False)
	(out, err) = trec_eval_res.communicate()
	trec_eval_res = out.decode("utf-8")
	print '\n'.join(trec_eval_res.split('\n')[5:6])
	file = open(eval_name, 'w')
	file.write(trec_eval_res)
	file.close()

def trec(golden_data, predictions_data, evalfile_name):
	temp_dir = uuid.uuid4().hex
	qrels_temp_file = '{0}/{1}'.format(temp_dir, 'qrels.txt')
	qret_temp_file = '{0}/{1}'.format(temp_dir, 'qret.txt')
	try:
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir)
		else:
			sys.exit("Possible uuid collision")

		format_bioasq2treceval_qrels(golden_data, qrels_temp_file)
		format_bioasq2treceval_qret(predictions_data, qret_temp_file)# qret_temp_file = '../Robust04.BM25b0.35_1.res'
		trec_evaluate(qrels_temp_file, qret_temp_file, evalfile_name)

	finally:
		os.remove(qrels_temp_file)
		os.remove(qret_temp_file)
		os.rmdir(temp_dir)

#128.62
if __name__ == "__main__":
	#loading vector
	Target ='robust04'#'Blog06','wt10g'
	print '************** Loading **************'

	'''# w2v_file = './robust04_data/rob04_w2v_200D_sg.bin'
	# w2v_file = './robust04_data/rob04_w2v_30D_sg.bin'
	# w2v_file = './GoogleNews-vectors-negative300.bin'
	
	# w2v_file = './vector_bins/glove.twitter.27B.200d.g.txt'
	# w2v_file = './vector_bins/glove.twitter.27B.100d.g.txt'
	# w2v_file = './vector_bins/glove.twitter.27B.50d.g.txt'
	# w2v_file = './vector_bins/glove.twitter.27B.25d.g.txt'

	# w2v_file = './vector_bins/glove.6B.300d.g.txt'
	# w2v_file = './vector_bins/glove.6B.200d.g.txt'
	# w2v_file = './vector_bins/glove.6B.100d.g.txt'
	# w2v_file = './vector_bins/'+Target+'_stem_100d.bin'
	# w2v_file = './vector_bins/glove.6B.50d.g.txt'
	# w2v_file = './vector_bins/glove.6B.50d.g.stem.txt'#'''
	w2v_file = './vector_bins/glove.6B.50d.g.robust04.newdocs.txt'

	Embedding = 50
	emd ='glove'

	global Stopwords
	Stopwords = list(set(stopwords.words('english'))-set(['ma','most','against','e','t','j','non']))

	weights = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.65,0.75,0.85,1.0]

	global w2v 
	if w2v_file.find('.bin')!= -1:
		w2v = KeyedVectors.load_word2vec_format(w2v_file, binary = True)#
	else:
		w2v = KeyedVectors.load_word2vec_format(w2v_file, binary = False)#
	print ("num words already in word2vec: " + str(len(w2v.vocab.keys())))

	global wind
	wind = 'winFlexLencoWeighted'
	if not os.path.exists('./results/{0}/{1}_top2/'.format(Target,wind)):
		os.makedirs('./results/{0}/{1}_top2/'.format(Target,wind))

	global Cons_1, Cons_2
	for Cons_1 in range(1,9):
		for Cons_2 in range(1,16):
			method ='{0}-qlen*gaussian-{1}-cons2-Max1-meanwid-Co15'.format(Cons_1+1,Cons_2)
			# method ='qry-tv-{0}-qrylen-{1}-Max1-Co15-meanwid--newqry'.format(Cons_1+1,Cons_2)
			e_5 = emd+'_'+str(Embedding)+'d_'+method+'.eval'#for results csv files
			global query_path
			global doc_path
			query_path = './Topics_Qrels/queries/new/orig/'+Target+'/'
			doc_path = './data/'+Target+'/new/docs/'
			qrets_bm25 = './'+Target+'.BM25b0.35.res'
			print '*****************Scoring **************'
			qid_doc_score = qid_docs_scores(qrets_bm25, wind, weights)

			print '************** Evaluation **************'
			qrels_file = './Topics_Qrels/qrels/qrels.'+Target+'.txt'
			qrels_bm25 = './'+Target+'.qrels'#BM25.WT2G.qrels
			if not os.path.exists(qrels_bm25):
				os.mknod(qrels_bm25)
				with open(qrels_bm25,'w') as fw:
					with open(qrels_file) as fin:
						for line in fin.readlines():
							items = line.strip().split(' ')
							if items[-1] != '0':
								fw.write(line)
				fw.close()
			prediction_data_bm25 = load_qrets(qrets_bm25)
			golden_data = load_qrels(qrels_bm25)
			res_dict = {'questions': []}
			for qid ,value in prediction_data_bm25.items():
				docnos = value['docno']
				bm25_scores = prediction_data_bm25[qid]['bm25']
				retr_scores = list(zip(docnos, bm25_scores))
				shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
				sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
				res_dict['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
			trec(golden_data, res_dict, './results/'+Target+'_bm25_0.35.eval')

			for idx,we in enumerate(weights):
				for bm25weighting in weights:
					predictions_data = Combine5_bm25(qid_doc_score, prediction_data_bm25, idx, bm25weighting,topk=1000)
					print 'processing : Windows_size %s, Max_2_weight%.2f, bm25_weight%.2f '%(str(wind),we,bm25weighting)
					system_name='window_top2'
					evalfile_name = './results/{0}/{1}_top2/{2}_Top2_{3}_bm25_{4}_{5}'\
					.format(Target,wind,wind,we,bm25weighting,e_5)
					trec(golden_data, predictions_data, evalfile_name)
					print#'''
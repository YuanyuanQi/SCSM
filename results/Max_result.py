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
import time
import random
import subprocess
import sys, re, os
import multiprocessing
import numpy as np
from scipy.special import comb,perm
from collections import OrderedDict
from gensim.models import KeyedVectors
from collections import defaultdict
from random import shuffle
from nltk.corpus import stopwords

for Target in ['robust04','blog06','wt10g']:
	with open(Target+'_winFlexLen_top2_results.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		Dict = {}
		num_q = 0
		for row in spamreader:
			if row[0] not in Dict:
				Dict[row[0]] = {'topk_wg':[],'win2_weighting':[],'bm25_wg':[],'map':[],'left':[]}
			Dict[row[0]]['topk_wg'].append(row[5])
			Dict[row[0]]['win2_weighting'].append(row[6])
			Dict[row[0]]['bm25_wg'].append(row[8])
			Dict[row[0]]['map'].append(row[10])
			Dict[row[0]]['left'].append(row[11:])

		csv_file = open(Target+'_winFlexLen_top2_results_MAX.csv','w')
		f_csv = csv.writer(csv_file)
		# Header =['method','width','embedding','dims','topk','topk_wg','win2_weighting','bm25','bm25_wg','num_q','map','Rprec','P_5','P_10','P_15','P_20','P_30','P_100','ndcg','ndcg_cut_5','ndcg_cut_10','ndcg_cut_15','ndcg_cut_20','ndcg_cut_30','ndcg_cut_100']
		# f_csv.writerow(Header)
		for key,value in Dict.items():
			maps = value['map']
			max_index = maps.index(max(maps))
			line = [key]
			line.extend(['FlexLen', 'glove', '50d', '2'])
			line.append(value['topk_wg'][max_index])
			line.append(value['win2_weighting'][max_index])
			line.append('bm25b0.35')
			line.append(value['bm25_wg'][max_index])
			line.append('num_q')
			line.append(max(maps))
			line.extend(value['left'][max_index])
			f_csv.writerow(line)
		csv_file.close()
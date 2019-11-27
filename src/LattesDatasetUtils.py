#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

import nltk
import pandas as pd

from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

class LattesDatasetUtils(object):
	"""
	Lattes Dataset Utils v 1.0.

	Parameter
	---------------------------
	description: str, optional (default='Lattes Dataset Utils v 1.0.')
		Lattes Dataset Normalization for Text Classification

	Attributes
	---------------------------
	description: str
		Lattes Dataset Utils v 1.0.

	parser: argparse
		Command line argument parser object
	"""
	def __init__(self, description='Lattes Dataset Normalization for Text Classification'):
		self.description = description
		self.parser = argparse.ArgumentParser(description=self.description)

		self.__init_parser__()

		self.parse_args()

		self.args.output = self.args.output.strip('/')

		if not os.path.exists(''.join([self.args.output, '/', self.args.set])):
			os.makedirs(''.join([self.args.output, '/', self.args.set]))

		self.__load__()

	def __init_parser__(self):
		self.parser.add_argument('dataset', type=str, help='Dataset filename')
		self.parser.add_argument('-s', '--set', type=str, help='Dataset word set format (Check IJDL article for more information)',
			default='ttw-set', choices=['ttw-set', 'etw-set'])
		self.parser.add_argument('-f', '--format', type=str, nargs='+', help='Output format list for text classification',
			default='text', choices=['text', 'tf', 'tfidf'])
		self.parser.add_argument('-o', '--output', type=str, help='Output file path', default='./')

	def __load__(self):
		print 'Loading dataset...'
		self.load_dataset()
		
		print 'Loading vectorizers...'
		self.load_vectorizer()
		
		print 'Loading mappers...'
		self.class_to_number_mapper = self.load_class_mapper()
		self.number_to_class_mapper = self.load_number_mapper()

	def parse_args(self):
		self.args = self.parser.parse_args()

	def load_dataset(self):
		self.data = pd.read_csv(self.args.dataset, sep='\t', header=0, na_values='NA').dropna()

	def load_vectorizer(self):
		if not os.path.exists(''.join([self.args.output, '/', self.args.set, '/tf_vectorizer.plk'])):
			self.tf_vectorizer = TfidfVectorizer(norm='l2', sublinear_tf=True, 
				stop_words=nltk.corpus.stopwords.words('portuguese'))
			self.tf_vectorizer.fit(self.data.title)
			joblib.dump(self.tf_vectorizer, ''.join([self.args.output, '/', self.args.set, '/tf_vectorizer.plk']))
		else:
			self.tf_vectorizer = joblib.load(''.join([self.args.output, '/', self.args.set,'/tf_vectorizer.plk']))

		if not os.path.exists(''.join([self.args.output, '/', self.args.set, '/tfidf_vectorizer.plk'])):
			self.tfidf_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True, 
				stop_words=nltk.corpus.stopwords.words('portuguese'))
			self.tfidf_vectorizer.fit(self.data.title)
			joblib.dump(self.tf_vectorizer, ''.join([self.args.output, '/', self.args.set, '/tfidf_vectorizer.plk']))
		else:
			self.tfidf_vectorizer = joblib.load(''.join([self.args.output, '/', self.args.set,'/tfidf_vectorizer.plk']))

	def load_class_mapper(self):
		if os.path.exists(''.join([self.args.output, '/', self.args.set, '/class_mapper.json'])):
			return json.load(open(''.join([self.args.output, '/', self.args.set, '/class_mapper.json']), 'r'))
		else:
			class_mapper = dict()
			class_mapper['1'] = dict()
			class_mapper['2'] = dict()
			class_mapper['3'] = dict()

			i = 1
			for ma in self.data.major_area.unique():
				class_mapper['1'][ma] = str(i)
				i += 1

			i = 0
			for a in self.data.area.unique():
				class_mapper['2'][a] = str(i)
				i += 1

			i = 0
			for sa in self.data.subarea.unique():
				class_mapper['3'][sa] = str(i)
				i += 1

			json.dump(class_mapper, open(''.join([self.args.output, '/', self.args.set, '/class_mapper.json']), 'w'))
			return class_mapper

	def load_number_mapper(self):
		if os.path.exists(''.join([self.args.output, '/', self.args.set, '/number_mapper.json'])):
			return json.load(open(''.join([self.args.output, '/', self.args.set, '/number_mapper.json']), 'r'))
		else:
			class_mapper = dict()
			class_mapper['1'] = dict()
			class_mapper['2'] = dict()
			class_mapper['3'] = dict()

			i = 1
			for ma in self.data.major_area.unique():
				class_mapper['1'][str(i)] = ma
				i += 1

			i = 1
			for a in self.data.area.unique():
				class_mapper['2'][str(i)] = a
				i += 1

			i = 1
			for sa in self.data.subarea.unique():
				class_mapper['3'][str(i)] = sa
				i += 1

			json.dump(class_mapper, open(''.join([self.args.output, '/', self.args.set, '/number_mapper.json']), 'w'))
			return class_mapper

	def number_to_class(self, number, level):
		return self.number_to_class_mapper[str(level)][str(number)]

	def class_to_number(self, _class_, level):
		return self.class_to_number_mapper[str(level)][_class_]

	def gen_text_folds(self):
		for level in [1, 2, 3]:
			views = list()
			if level == 1:
				tag = 'major_area'
				views.append((tag, self.data))
			elif level == 2:
				tag = 'area'
				for _class_ in self.data.major_area.unique():
					views.append((_class_, self.data[self.data.major_area == _class_]))
			elif level == 3:
				tag = 'subarea'
				for _class_ in self.data.area.unique():
					views.append((_class_, self.data[self.data.area == _class_]))
			else:
				print 'Invalid level, must pass one of the following: [1: major areas, 2: areas, 3: subareas]'
				return

			if tag is not None:
				for view in views:
					if view[1].shape[0] < 10:
						print 'Couldn\'t create train and test folds for {0}. Small # of instances.'.format(view[0])
					else:
						output_path = ''.join([self.args.output, '/', self.args.set, '/level_', `level`, '/', view[0].replace(' ', '_')])
						if not os.path.exists(output_path):
							os.makedirs(output_path)

						if not os.path.exists(''.join([output_path, '/data/text'])):
							os.makedirs(''.join([output_path, '/data/text']))

						print '------------------'

						kf = KFold(n_splits=5, random_state=42, shuffle=True)
						k = 1
						for train_ix, test_ix in kf.split(view[1]['title'], view[1][['major_area', 'area', 'subarea']]):
							print 'Generating text fold {0} for {1}'.format(k, view[0])

							# Split data into subsets train and test
							train = view[1].iloc[train_ix]
							test = view[1].iloc[test_ix]

							train.to_csv(''.join([output_path, '/data/text/train_fold', `k`, '.csv']), sep='\t')
							test.to_csv(''.join([output_path, '/data/text/test_fold', `k`, '.csv']), sep='\t')

							k += 1

	def gen_tf_folds(self):
		for level in [1, 2, 3]:
			output_path = ''.join([self.args.output, '/', self.args.set, '/level_', `level`, '/'])
			folders = list()
			folders.extend([name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))])
			for folder in folders:

				print '------------------'

				for i in range(1, 6):
					print 'Generating TF fold {0} for {1}'.format(i, folder.replace('_', ' '))

					if not os.path.exists(''.join([output_path, folder, '/data/tf'])):
						os.makedirs(''.join([output_path, folder, '/data/tf']))

					train = pd.read_csv(''.join([output_path, folder, '/data/text/train_fold', `i`, '.csv']), sep='\t', header=0)
					train.major_area = train.major_area.apply(lambda x: self.class_to_number(x, 1))
					train.area = train.area.apply(lambda x: self.class_to_number(x, 2))
					train.subarea = train.subarea.apply(lambda x: self.class_to_number(x, 3))
					X_train = self.tf_vectorizer.transform(train.title)
					y_train = train[['major_area', 'area', 'subarea']]
					dump_svmlight_file(X_train, y_train, f=''.join([output_path, folder, '/data/tf/train_fold', `i`, '.svm']), multilabel=True)

					test = pd.read_csv(''.join([output_path, folder, '/data/text/test_fold', `i`, '.csv']), sep='\t', header=0)
					test.major_area = test.major_area.apply(lambda x: self.class_to_number(x, 1))
					test.area = test.area.apply(lambda x: self.class_to_number(x, 2))	
					test.subarea = test.subarea.apply(lambda x: self.class_to_number(x, 3))
					X_test = self.tf_vectorizer.transform(test.title)
					y_test = test[['major_area', 'area', 'subarea']]
					dump_svmlight_file(X_test, y_test, f=''.join([output_path, folder, '/data/tf/test_fold', `i`, '.svm']), multilabel=True)			

	def gen_tfidf_folds(self):
		for level in [1, 2, 3]:
			output_path = ''.join([self.args.output, '/', self.args.set, '/level_', `level`, '/'])
			folders = list()
			folders.extend([name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))])
			for folder in folders:

				print '------------------'

				for i in range(1, 6):
					print 'Generating TF-IDF fold {0} for {1}'.format(i, folder.replace('_', ' '))

					if not os.path.exists(''.join([output_path, folder, '/data/tfidf'])):
						os.makedirs(''.join([output_path, folder, '/data/tfidf']))

					train = pd.read_csv(''.join([output_path, folder, '/data/text/train_fold', `i`, '.csv']), sep='\t', header=0)
					train.major_area = train.major_area.apply(lambda x: self.class_to_number(x, 1))
					train.area = train.area.apply(lambda x: self.class_to_number(x, 2))
					train.subarea = train.subarea.apply(lambda x: self.class_to_number(x, 3))
					X_train = self.tfidf_vectorizer.transform(train.title)
					y_train = train[['major_area', 'area', 'subarea']]
					dump_svmlight_file(X_train, y_train, f=''.join([output_path, folder, '/data/tfidf/train_fold', `i`, '.svm']), multilabel=True)

					test = pd.read_csv(''.join([output_path, folder, '/data/text/test_fold', `i`, '.csv']), sep='\t', header=0)
					test.major_area = test.major_area.apply(lambda x: self.class_to_number(x, 1))
					test.area = test.area.apply(lambda x: self.class_to_number(x, 2))	
					test.subarea = test.subarea.apply(lambda x: self.class_to_number(x, 3))
					X_test = self.tfidf_vectorizer.transform(test.title)
					y_test = test[['major_area', 'area', 'subarea']]
					dump_svmlight_file(X_test, y_test, f=''.join([output_path, folder, '/data/tfidf/test_fold', `i`, '.svm']), multilabel=True)

	def run(self):
		for _format_ in self.args.format:
			if _format_ == 'text':
				print 'TEXT FOLDS GENERATOR METHOD RUNNING'
				self.gen_text_folds()
				print '______________________________'
			elif _format_ == 'tf':
				print 'TF FOLDS GENERATOR METHOD RUNNING'
				self.gen_tf_folds()
				print '______________________________'
			elif _format_ == 'tfidf':
				print 'TF-IDF FOLDS GENERATOR METHOD RUNNING'
				self.gen_tfidf_folds()
				print '______________________________'
			else:
				print 'ERROR: Invalid format'


if __name__ == '__main__':
	app = LattesDatasetUtils()
	app.run()





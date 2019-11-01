import pandas as pd
import numpy as np
import os
from os.path import exists,abspath
from os import mkdir, makedirs
from sys import argv
from glob import glob
from multiprocessing import Pool
from itertools import product
from tqdm import tqdm
import networkx as nx
import argparse
import pdb
import shutil
import pickle


def convert_to_arff(df,path):
    with open(path, 'w') as fn:
        fn.write('@relation linhuaw\n')
        for col in df.columns:
            if col != 'cls':
                fn.write('@attribute %s numeric\n' %col)
            else:
                fn.write('@attribute cls {-1.0,1.0}\n')
        fn.write('@attribute seqID string\n')
        fn.write('@data\n')
        for i in range(df.shape[0]):
            fn.write(','.join(map(str,df.values[i])))
            fn.write(',%s\n' % df.index[i])


def processTermFeature(param, out_dir, min_num_pos=5, non_edge_list=False, pickle_node_ids=None):
    """
    *min_num_pos*: keep a feature only if it has at least X positives. Default=5
    """
    term, feature, feature_file = param
    t = term[:2] + term[3:]
    # I think this may be an adjacency matrix
    print("reading %s" % (feature_file))
    # for now, treat the network file as an edge list, and convert to an adjacency matrix
    #feature_df =  pd.read_csv(net_file,index_col=0, sep=',')
    if non_edge_list and pickle_node_ids is not None:
        #deepNF stores the features as a numpy adjacency matrix 
        features = pickle.load(open(feature_file, 'rb'))
        print(features.shape)
        node_ids = pd.read_csv(pickle_node_ids, header=None, index_col=None, squeeze=True)
        print(node_ids.head())
        print(node_ids.values)
        print("%d node ids" % len(node_ids))
        # remove self loops by setting the diagonal to 0s
        features = features - np.diag(np.diag(features)) 
        feature_df = pd.DataFrame(features, index=node_ids.values)
        print(feature_df.head())
    elif non_edge_list:
        feature_df = pd.read_csv(feature_file, sep='\t')
    else:
        feature_df = pd.read_csv(feature_file, sep='\t')
        #print(feature_df.head())
        print("converting to adjacency matrix")
        feature_df.columns = ["node1", "node2", 'weight']
        # divide by 1000 to get the weight
        # no longer needed as I pre-divided the weights
        #feature_df['weight'] = feature_df['weight'].apply(lambda x: x / 1000.0)
        G = nx.from_pandas_edgelist(feature_df,'node1','node2','weight')
        print("\t%d nodes, %d edges" % (G.number_of_nodes(), G.number_of_edges()))
        feature_df = nx.to_pandas_adjacency(G)
    #print(feature_df.head())
    before_shape = feature_df.shape
    binary_term = binary_table[term]
    # get the positive and negative examples for this term
    binary_term = binary_term.loc[binary_term != 0]
    term_inds = binary_term.index.tolist()
    print("\t%d positive and negative examples" % (len(term_inds)))
    # get the rows corresponding to the positive and negative examples
    sel_inds = [ind for ind in feature_df.index.tolist() if ind in term_inds]
    print("\t%d corresponding rows in the network" % (len(sel_inds)))
    feature_df = feature_df.loc[sel_inds,]
    binary_term = binary_term.loc[sel_inds]
    # now remove the columns that have all 0s
    cols = (feature_df == 0).all(axis=0)
    cols = cols.loc[cols == False].index.tolist()
    feature_df = feature_df[cols]

    print("term: %s, feature: %s, before_shape: %s, after shape: %s" % (
        term,feature,before_shape,feature_df.shape))

    # if there's no edges to write, then skip this network
    if feature_df.shape[1] == 0:
        print("no edges to write. skipping")
        return

    feature_df['cls'] = binary_term
    # or if there's less than X positives, then skip this network
    num_pos_rows = len(feature_df[feature_df['cls'] == 1])
    if num_pos_rows < min_num_pos:
        print("%d positive rows < %d minimum required. skipping" % (num_pos_rows, min_num_pos))
        return

    del feature_df.index.name
    p = '%s/%s' %(out_dir,feature)
    if not exists(p):
        makedirs(p)
    #path = p + '/' + t + '.arff'
    path = p + '/' + 'data.arff'
    print("writing %s" % (path))
    convert_to_arff(feature_df,path)


if __name__== "__main__":
    #data_dir = abspath('/sc/orga/scratch/wangl35/info_content/combined_string')
    ### parse arguments
    parser = argparse.ArgumentParser(description='Preprocess the input files')
    parser.add_argument('--feature-dir', '-F', type=str, required=True, 
            help='path to directory containing .csv.gz feature files (e.g., string adjacency matrices)')
    parser.add_argument('--pos-neg-file', '-A', type=str, required=True, 
            help='Table of positive, negative and unknown assignments with terms as the rows, and genes in the columns')
    parser.add_argument('--weka-path', '-W', type=str, 
            help='path to dir with weka files "weka.properties" and "classifiers.txt" to copy to --out-path. Only copied if specified')
    parser.add_argument('--term', '-T', type=str, required=True, 
            help='Term for which to create the .arff file')
    parser.add_argument('--out-path', '-P', type=str, required=True, 
            help='path to output directory')
    parser.add_argument('--pickle-node-ids', type=str, 
            help='Option to use *.pckl files instead of *.csv.gz files. Give the path to a file with the node ids')
    parser.add_argument('--non-edge-list-features', action='store_true', default=False,
            help='If the .csv.gz feature files are not edge lists, then use this option')
    parser.add_argument('--forced', '-f', action='store_true', default=False,
            help='force overwriting the output files')
    args = parser.parse_args()

    #data_dir = abspath('inputs/2017_10-string-nontrans/')
    term = args.term
    t = term[:2] + term[3:]
    out_dir = "%s/%s" % (args.out_path, t)
    if not exists(out_dir):
        makedirs(out_dir)
    if args.weka_path is not None:
        for f in ["weka.properties", "classifiers.txt"]:
            w_f = "%s/%s" % (args.weka_path, f)
            out_file = "%s/%s" % (out_dir, f)
            if args.forced or not os.path.isfile(out_file):
                print("copying %s to %s" % (w_f, out_file))
                shutil.copy(w_f, out_dir)
    if args.pickle_node_ids is not None:
        feature_files = glob('%s/*.pckl' % (args.feature_dir))
    else:
        feature_files = glob('%s/*.csv.gz' % (args.feature_dir))
    features = [f.split('/')[-1].split('.')[0] for f in feature_files]
#    features = ['combined_score']
    binary_table = pd.read_table(args.pos_neg_file,sep='\t',index_col=0)

    p = Pool(13)
    params = [(term, features[i], feature_files[i]) for i in range(len(features))]
    print(params)
    for param in tqdm(params):
        term, feature, feature_file = param
        t = term[:2] + term[3:]
        arff_file = '%s/%s/data.arff' %(out_dir,feature)
        if args.forced is False and os.path.isfile(arff_file):
            print("%s already exists. Use --forced to overwrite" % (arff_file))
            continue
        processTermFeature(param, out_dir, 
                non_edge_list=args.non_edge_list_features,
                pickle_node_ids=args.pickle_node_ids,
                )



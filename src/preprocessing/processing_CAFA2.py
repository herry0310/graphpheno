
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--species', type=str, default="CAFA2", help="which species to use.")
args = parser.parse_args()
print("Start processing uniprot...")

#### load file
print("Loading data...")
uniprot = pd.read_table(os.path.join(args.data_path, args.species, "uniprot.txt"), delimiter="\t")
print(uniprot.shape)

#### filtering
print("filtering...")
# filter by STRING ID occurence
uniprot = uniprot[~uniprot['Cross-reference (STRING)'].isna()]
uniprot.index = range(uniprot.shape[0])
uniprot['Cross-reference (STRING)'] = uniprot['Cross-reference (STRING)'].apply(lambda x:x[:-1])

# filter by ambiguous amino acid
def find_amino_acid(x):
    return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)

ambiguous_index = uniprot.loc[uniprot['sequences'].apply(find_amino_acid)].index
uniprot.drop(ambiguous_index, axis=0, inplace=True)
uniprot.index = range(len(uniprot))
print("after filtering:", uniprot.shape)

#### encode amino acid sequence using CT
print("encode amino acid sequence using CT...")
def CT(sequence):
    classMap = {'G':'1','A':'1','V':'1','L':'2','I':'2','F':'2','P':'2',
            'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
            'R':'5','K':'5','D':'6','E':'6','C':'7'}

    seq = ''.join([classMap[x] for x in sequence])
    length = len(seq)
    coding = np.zeros(343,dtype=np.int)
    for i in range(length-2):
        index = int(seq[i]) + (int(seq[i+1])-1)*7 + (int(seq[i+2])-1)*49 - 1
        coding[index] = coding[index] + 1
    return coding

CT_list = []
for seq in uniprot['sequences'].values:
    CT_list.append(CT(seq))
uniprot['CT'] = CT_list


uniprot.to_pickle(os.path.join(args.data_path, args.species, "features.pkl"))
#################################
######## process PPIs ###########

print("Start processing PPIs...")
string = pd.read_table(os.path.join(args.data_path, args.species, "string_9.05_human.txt"), delimiter=" ")
gene_list = uniprot[['Entry name','Cross-reference (STRING)']]

# filter by uniprot
string = string[string['protein1'].isin(gene_list['Cross-reference (STRING)'].values)]
string = string[string['protein2'].isin(gene_list['Cross-reference (STRING)'].values)]

# map names to indexs
id_mapping = dict(zip(list(gene_list['Cross-reference (STRING)'].values),
                      list(gene_list.index)))
                     # list(gene_list['Entry name'].values)))
string['protein1_id'] = string['protein1'].apply(lambda x:id_mapping[x])
string['protein2_id'] = string['protein2'].apply(lambda x:id_mapping[x])

subnetwork = string[['protein1_id','protein2_id','combined_score']]
subnetwork['combined_score'] = subnetwork['combined_score']/1000.0
subnetwork.to_csv(os.path.join(args.data_path, args.species, "ppi.txt"), index=False, header=False, sep="\t")



# test = pd.read_table(os.path.join(args.data_path, args.species, "test.txt"), delimiter="\t")
# train = pd.read_table(os.path.join(args.data_path, args.species, "train.txt"), delimiter="\t")
train = pd.read_pickle(os.path.join(args.data_path, args.species, "human.pkl"))
test = pd.read_pickle(os.path.join(args.data_path, args.species, "human_test.pkl"))
uniprot_label = pd.merge(uniprot,train[['proteins','hp_annotations']],left_on="Entry name",right_on='proteins',how ="inner")
uniprot_pre = pd.merge(uniprot,test[['proteins','hp_annotations']],left_on="Entry name",right_on='proteins',how ="inner")
uniprot_label.drop(['proteins'],axis=1,inplace=True)
uniprot_pre.drop(['proteins'],axis=1,inplace=True)
#index
uniprot_label_index = uniprot[uniprot['Entry name'].isin(train['proteins'].values)].index
uniprot_pre_index = uniprot[uniprot['Entry name'].isin(test['proteins'].values)].index

print("obtain HP annotations...")
items = [item for sublist in uniprot_label['hp_annotations'] for item in sublist]
unique_elements, counts_elements = np.unique(items, return_counts=True)
hp_list_t = unique_elements[np.where(counts_elements >= 10)]#CAFA_train
temp_hp = uniprot_pre['hp_annotations'].apply(lambda x: list(set(x) & set(hp_list_t)))
items_test = [item for sublist in temp_hp for item in sublist]
unique_elements_test, counts_elements_test = np.unique(items_test, return_counts=True)
hp_list = unique_elements_test[np.where(counts_elements_test >= 1)]#CAFA_test
#表型筛选
temp_hp = uniprot_label['hp_annotations'].apply(lambda x: list(set(x) & set(hp_list)))
uniprot_label['filter_hp'] = temp_hp
num_hp = sum(len(x) for x in uniprot_label['filter_hp'])
print("number of hpo", num_hp)

temp_hp = uniprot_pre['hp_annotations'].apply(lambda x: list(set(x) & set(hp_list)))
uniprot_pre['filter_hp'] = temp_hp


# write out filtered ontology lists
def write_hp_list(ontology,ll):
    filename = os.path.join(args.data_path, args.species, ontology+"_list.txt")
    with open(filename,'w') as f:
        for x in ll:
            f.write(x + '\n')
print("writing hp term list...")
write_hp_list('hp',hp_list)


#### encode HP terms
print("encoding HP terms...")
hp_dict = dict(zip(list(hp_list),range(len(hp_list))))
hp_encoding = [[0]*len(hp_dict) for i in range(len(uniprot_label))]
hp_encoding_test = [[0]*len(hp_dict) for i in range(len(uniprot_pre))]

for i,row in uniprot_label.iterrows():
    for x in row['filter_hp']:
        hp_encoding[i][ hp_dict[x] ] = 1

uniprot_label['hp_label'] = hp_encoding
for i,row in uniprot_pre.iterrows():
    for x in row['filter_hp']:
        hp_encoding_test[i][ hp_dict[x] ] = 1

uniprot_pre['hp_label'] = hp_encoding_test

uniprot_pre.to_pickle(os.path.join(args.data_path, args.species, "features_pre.pkl"))
uniprot_label.to_pickle(os.path.join(args.data_path, args.species, "features_label.pkl"))
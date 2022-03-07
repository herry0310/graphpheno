from collections import deque
import pandas as pd
from xml.etree import ElementTree as ET

def get_phenotype_ontology(filename='../../data/hp.obo.txt'):
    # Reading Gene Ontology from OBO Formatted file
    hp = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    hp[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        hp[obj['id']] = obj
    for hp_id in list(hp.keys()):
        if hp[hp_id]['is_obsolete']:
            del hp[hp_id]
    for hp_id, val in hp.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in hp:
                if 'children' not in hp[p_id]:
                    hp[p_id]['children'] = set()
                hp[p_id]['children'].add(hp_id)
    return hp


def get_anchestor(hp, hp_id):
    hp_set = set()
    q = deque()
    q.append(hp_id)
    while(len(q) > 0):
        g_id = q.popleft()
        hp_set.add(g_id)
        if g_id not in hp:
            #print g_id
            continue
        for parent_id in hp[g_id]['is_a']:
            if parent_id in hp:
                q.append(parent_id)
    return hp_set


def get_parents(hp, hp_id):
    hp_set = set()
    for parent_id in hp[hp_id]['is_a']:
        if parent_id in hp:
            hp_set.add(parent_id)
    return hp_set


def get_hp_set(hp, hp_id):
    hp_set = set()
    q = deque()
    q.append(hp_id)
    while len(q) > 0:
        g_id = q.popleft()
        hp_set.add(g_id)
        for ch_id in hp[g_id]['children']:
            q.append(ch_id)
    return hp_set
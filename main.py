from json.tool import main
from unicodedata import name
import torch
from sklearn.metrics import classification_report
from data_config import get_path
from tpipeline import TPipeline,TaggerDataset
import time
import argparse
print("hello")
def acc(path,test_d_path):
    f = open(test_d_path,'r')
    gold =  f.readlines()
    f.close()
    f = open(path)
    pred =  f.readlines()
    f.close()
    w = open('combine.pks.conll','w')
    w.write('word_id	word	postag	lemma	gold_head	gold_label	pred_head	pred_label\n')
    for i in range(len(gold)):
        if gold[i] == '\n':
            w.write('\n')
            continue
        gold[i] = gold[i].split('\t')
        gold[i][-1] = gold[i][-1].replace('\n','')
        pred[i] = pred[i].split('\t')
        pred[i][-1] = pred[i][-1].replace('\n','')
        temp = [gold[i][0],gold[i][1],gold[i][3],gold[i][3],gold[i][6],gold[i][7],pred[i][6],pred[i][7]]
        w.write('\t'.join(temp)+'\n')
    w.close()
    targs = []
    preds= []
    pr,tg=[],[]
    # print(pred)
    for i in range(len(pred)):
        if gold[i] == '\n':
            continue
        preds.append(pred[i][7])
        targs.append(gold[i][7])
    target_names = list(set(targs))
    print(classification_report(preds, targs, target_names=target_names))
    f = open('eval_matrix.txt','w')
    f.write(str(classification_report(preds, targs, target_names=target_names)))
    f.close()

def run(panelty,model_path,train_path,dev_path,test_d_path):
    torch.cuda.empty_cache()
    trainer = TPipeline(
            training_config={
            'category': 'customized-mwt-ner', # pipeline category
            'task': 'posdep', # task name
            'save_dir': model_path, # directory for saving trained model
            'train_conllu_fpath': train_path, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
            'max_epoch': 1,
            "batch_size":30,
            'panelty':panelty
        })
    trainer.train()
    test_set = TaggerDataset(
        config=trainer._config,
        input_conllu=test_d_path,
        gold_conllu=test_d_path,
        evaluate=True
    )
    test_set.numberize()
    test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
    result,path = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                            name='testfaL', epoch=-1)
    print(path)
    del trainer
    torch.cuda.empty_cache()
    acc(path,test_d_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='.', help='Model path')
    parser.add_argument('--experiment', type=str, default='sactii fine', help='Experiment type')
    args = parser.parse_args()
    exp_type = args.experiment
    print("gn ",exp_type)

    train_path,dev_path,test_path = get_path(exp_type)
    model_path = args.model_path #'./models/'
    
    
    panelty = 0.01
    run(panelty,model_path,train_path,dev_path,test_path)




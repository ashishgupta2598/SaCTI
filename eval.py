
#[0.005,0.003,0.001,0.0001]
tp = 'test'#test
task = 'grammer_swap0.6'
vals = 'main'
path_exp ='/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/'+tp+'_grammer.conll'

f = open(path_exp,'r')
gold =  f.readlines()
f.close()
#f = open("/home/kabira/Documents/githubs/save_dir_trankit/grammer_swap0.8/xlm-roberta-base/customized-mwt-ner/preds/tagger.testfaL.conllu.epoch--1")
gk = 'tagger.testfaL.conllu.epoch--1' if tp=='test' else 'tagger.dev.conllu'
print(gk)
ash ='/home/kabira/Documents/githubs/save_dir_trankit/'+task+'/xlm-roberta-base/customized-mwt-ner/preds/'+gk
f = open(ash)
# f = open('/home/kabira/Documents/githubs/save_dir_trankit/grammer_swap0.01/xlm-roberta-base/customized-mwt-ner/preds/tagger.testfaL.conllu.epoch--1')
pred =  f.readlines()
f.close()

w = open('combine.pks.conll','w')
print("gold file ",path_exp)
print('pred file ',ash)
# print('')
w.write('word_id	word	postag	lemma	gold_head	gold_label	pred_head	pred_label\n')
for i in range(len(gold)):
    try:
        if gold[i] == '\n':
            w.write('\n')
            continue
        gold[i] = gold[i].split('\t')
        gold[i][-1] = gold[i][-1].replace('\n','')
        pred[i] = pred[i].split('\t')
        pred[i][-1] = pred[i][-1].replace('\n','')
        if vals=="main":
            print("hello")
            temp = [gold[i][0],gold[i][1],gold[i][3],gold[i][3],gold[i][6],gold[i][7],pred[i][6],pred[i][7]]
        else:
            temp = [gold[i][0],gold[i][1],gold[i][3],gold[i][3],gold[i][9],gold[i][8],pred[i][9],pred[i][8]]

        w.write('\t'.join(temp)+'\n')
    except:
        import traceback
        traceback.print_exc()
        print("test sent error ",gold[i])
        print("pred sent error ",pred[i],i)
        break
    # if i==14:
    #     break
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

    

target_names = ['class 0', 'class 1', 'class 2','class 3']
import types
from sklearn.metrics import classification_report
train_type = 'sl'
types='glk'
print(classification_report(preds, targs, target_names=target_names))
f = open(train_type+'_'+types+'_eval_matrix.txt','w')
f.write(str(classification_report(preds, targs, target_names=target_names)))
f.close()

from Trakit_macro_UAS_LAS import run_eval
run_eval('eval_matrix.txt')
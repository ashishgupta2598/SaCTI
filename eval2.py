#[0.005,0.003,0.001,0.0001]
#grammer_swap0.005morph_grammer_main_task
#path_exp = '/home/kabira/Documents/ai-Compound-Classification/shffle context comparision task/posdep shuffle data/dev_normalshuffle.conll'
path_exp = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/morph_grammer_data/dev_morph.conll'
f = open(path_exp,'r')
gold =  f.readlines()
f.close()
tp = 'grammer_swap0.001morph_grammer_main_task'
ash  = '/home/kabira/Documents/githubs/save_dir_trankit/grammer_swapmorph_grammer_main_task/xlm-roberta-base/customized-mwt-ner/preds/tagger.testfaL.conllu.epoch--1'
ash  ='/home/kabira/Documents/githubs/save_dir_trankit/'+tp+'/xlm-roberta-base/customized-mwt-ner/preds/tagger.dev.conllu'
f = open(ash)
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
        #temp = [gold[i][0],gold[i][1],gold[i][3],gold[i][3],gold[i][6],gold[i][7],pred[i][6],pred[i][7]]
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
from cgi import test
train_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/onlygrammermain/train_onlygrammer.conll'#'/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/train_grammer.conll'
dev_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/onlygrammermain/dev_onlygrammer.conll'#'/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/dev_grammer.conll'
test_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/onlygrammermain/test_onlygrammer.conll'#'/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/test_grammer.conll'
train_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/train_grammer.conll'
dev_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/dev_grammer.conll'
test_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/test_grammer.conll'
train_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/morph_grammer_data/train_morph.conll'
dev_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/morph_grammer_data/dev_morph.conll'
test_d_path = '/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/morph_grammer_data/test_morph.conll'
test_d_path = '/home/kabira/Desktop/dk.conll'
from iterators.tagger_iterators import TaggerDataset
import torch
def acc(tp,task):
    path_exp ='/home/kabira/Documents/ai-Compound-Classification/postdep/task2 data with complete info/grammer/'+tp+'_grammer.conll'
    f = open(path_exp,'r')
    gold =  f.readlines()
    f.close()
    gk = 'tagger.testfaL.conllu.epoch--1' if tp=='test' else 'tagger.dev.conllu'
    ash ='/home/kabira/Documents/githubs/save_dir_trankit/'+task+'/xlm-roberta-base/customized-mwt-ner/preds/'+gk
    f = open(ash)
    pred =  f.readlines()
    f.close()
    w = open('combine.pks.conll','w')
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
            temp = [gold[i][0],gold[i][1],gold[i][3],gold[i][3],gold[i][6],gold[i][7],pred[i][6],pred[i][7]]
            #temp = [gold[i][0],gold[i][1],gold[i][3],gold[i][3],gold[i][9],gold[i][8],pred[i][9],pred[i][8]]

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
    print(classification_report(preds, targs, target_names=target_names))
    f = open(tp+'_'+task+'_eval_matrix.txt','w')
    f.write(str(classification_report(preds, targs, target_names=target_names)))
    f.close()
    from Trakit_macro_UAS_LAS import run_eval
    run_eval('eval_matrix.txt')

def run(panelty,morek):
    torch.cuda.empty_cache()
    from tpipeline import TPipeline
    trainer = TPipeline(
            training_config={
            'category': 'customized-mwt-ner', # pipeline category
            'task': 'posdep', # task name
            'save_dir': '/home/kabira/Documents/githubs/save_dir_trankit/'+'grammer_swap'+str(0.005)+morek, # directory for saving trained model
            'train_conllu_fpath': dev_d_path, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': dev_d_path, # annotations file in CONLLU format for development
            'max_epoch': 60,
            "batch_size":50,
            'panelty':panelty
        })

    # #
    # trainer.train()
    # exit()
    # print("*************** is ",test_d_path)
    test_set = TaggerDataset(
        config=trainer._config,
        input_conllu=test_d_path,
        gold_conllu=test_d_path,
        evaluate=True
    )
    import pickle
    # with open("test_data.pkl", 'wb') as pickle_file:
    #     pickle.dump(test_set,pickle_file)

    test_set.numberize()
    test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
    result,path = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                            name='testfaL', epoch=-1,task='test')
    # print(result)
    print(path)
    del trainer
    torch.cuda.empty_cache()

    # acc('test',task='grammer_swap'+str(panelty))
    # acc('dev',task='grammer_swap'+str(panelty))
import time
for pan in [0.01]:
    tt = time.time()
    run(pan,"morph_grammer_main_task")
    print('****************************TIME TO RUN******',str(pan),'*************************',time.time()-tt)
    break
    # time.sleep(10)

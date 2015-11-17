import subprocess
import os
import numpy as np


def sigmoid(x) :
    return 1/(1+np.exp(-x))

def myMakeDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class RGF_Model() :

    def_params = {}
    def_params['algorithm'] = 'RGF'
    def_params['reg_L2'] = 1e-10
    def_params['reg_sL2'] = def_params['reg_L2']/100 
    def_params['loss'] = 'Log'
    def_params['test_interval'] = 1000
    def_params['max_leaf_forest'] = 3000
    def_params['min_pop'] = 1

    def __init__(self, rgf_path, tmp_path, modelname = 'myfirstmodel', save_path = None, params = def_params, verbose = True):
        self.rgf_path= rgf_path
        self.tmp_path = tmp_path
        self.modelname = modelname
        if save_path :
            self.save_path = save_path
        else :     
            self.save_path = tmp_path + '/savedir'
        self.data_path = self.tmp_path + '/' + 'train_data'
        self.output_path = self.save_path + '/' + self.modelname 
        self.output_path_models = self.output_path + '/models'
        self.output_path_preds = self.output_path + '/preds'
        self.test_path = self.tmp_path + '/test_data'
        self.params = params  
        self.verbose = verbose
        myMakeDir(self.data_path)
        myMakeDir(self.output_path)
        myMakeDir(self.output_path_models)
        myMakeDir(self.output_path_preds)
        myMakeDir(self.test_path)
        # state variables
        self.trained = False
        self.predictions_made = False
        self.train_written = False
        self.test_written = False
        self.last_model = None
        # classes temp hack
        self.classes_ = [0,1]

        
    def fit(self, X, y) :

        # remap : assume you give 0,1 and we want -1, 1
        y = 2 *y -1

        # write data
        if self.verbose : 
            print "writing data"
        np.savetxt(self.data_path + '/trainx.txt', X, delimiter=' ')
        np.savetxt(self.data_path + '/trainy.txt', y, delimiter=' ')
        if self.verbose : 
            print "writing done"
            
        # write params
        with open (self.data_path+'/'+self.modelname+'.inp', 'w') as fp:        
            fp.write('train_x_fn=' + self.data_path + '/trainx.txt\n')
            fp.write('train_y_fn=' + self.data_path + '/trainy.txt\n')
            fp.write('model_fn_prefix=' + self.output_path_models + '/m\n')
            for p in self.params.items():
                fp.write("%s=%s\n" % p)
            fp.write('Verbose')

        # start RGF
        thecommand = 'perl ' + self.rgf_path + '/test/call_exe.pl ' \
                    + self.rgf_path + '/bin/rgf train ' \
                    + self.data_path + '/' + self.modelname
        p = subprocess.Popen(thecommand,
                             shell=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while p.poll() is None:
            if self.verbose :
                output = p.stdout.readline()
                print output   
                
        # get last model        
        files = os.listdir(self.output_path_models)
        self.last_model = np.sort(files)[-1]
        
    
    def build_predictions(self,X) :
        # write data
        if self.verbose : 
            print 'writing test data'
        np.savetxt(self.test_path + '/testx.txt', X, delimiter=' ')
        if self.verbose : 
            print 'test data written'
    
        # this is my dir 
        files = os.listdir(rgf.output_path_models) 
        for thefile in files : 
            if thefile.startswith("m-") : 
                with open (self.tmp_path+'/temp_pred.inp', 'w') as fp:
                    predsfile = os.path.join(self.output_path_preds, thefile)
                    modelfile = os.path.join(self.output_path_models, thefile)
                    fp.write('test_x_fn=' + self.test_path + '/testx.txt\n')
                    fp.write('model_fn=' + modelfile + '\n')
                    fp.write('prediction_fn=' + predsfile + '.pred')
                    fp.close()
                    command = 'perl ' + self.rgf_path + '/test/call_exe.pl ' + self.rgf_path + \
                              '/bin/rgf predict ' + self.tmp_path + '/temp_pred'
                    p = subprocess.Popen(command,shell=True, bufsize=1, stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT)
                    while p.poll() is None:
                        output = p.stdout.readline()
                        print output                
    
    def predict_proba(self,X, keeplast = True) :        
        
        self.build_predictions(X)
        
        if keeplast :
            with open(self.output_path_preds + '/' + self.last_model + '.pred','rb') as thefile:
                mypreds = sigmoid(np.loadtxt(thefile))
                mypreds = np.vstack((1-mypreds,mypreds)).T
        else : 
            for files in os.listdir(self.output_path_models) :
                pass
                # TODO
            
        return mypreds

        
    def predict(self, X, retrain = True, keeplast = True) :

        self.build_predictions(X)
    
        if keeplast :
            with open(self.output_path_preds + '/' + self.last_model + '.pred','rb') as thefile:
                mypreds = (sigmoid(np.loadtxt(thefile)) >= 0.5).astype(int) # only work for classif
        else : 
            for files in os.listdir(self.output_path_models) :
                pass
                # TODO
            
        return mypreds

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:25:35 2012

@author: josip
"""
import pandas
import numpy as np
from ffnet import ffnet,mlgraph,savenet
from bunch import Bunch
import threading
import pp
import os
import time

class Ensemble(object):
    """
    Klasa koja bi trebala imati podatke (pandas.DataFrame) i jednu ili
    više neuralnih mreža nad tim podacima. Prema van se ponaša kao jedan model,
    tj. sve neuralne mreže imaju iste ulaze i izlaze, a vrijednost se računa
    kao srednja vrijednost izlaza svih pojedinih mreža (ili većina ako je u
    pitanju klasifikcaija).
    Pokušat ću napraviti tako da nije ovisna o konkretnom modulu koji računa
    neuralne mreže - trenutno je to samo ffnet.
    """


    _df=None
    def _get_df(self): return self._df
#    def _set_df(self,value):
#        self._df=value
#        if self.invars != None and self.outvars != None:
#            self.set_vars(self.invars,self.outvars)
#        self.shuffle()  # moram ponovo izračunati random array ...
    df=property(_get_df,doc="DataFrame with data")

    _invars=None;
    def _get_invars(self): return self._invars
    def _set_invars(self,value): self.setvars(input_vars=value)
    invars=property(_get_invars,_set_invars,doc='Input variables')

    _outvars=None;
    def _get_outvars(self): return self._outvars
    def _set_outvars(self,value): self.setvars(target_vars=value)
    outvars=property(_get_outvars,_set_outvars,doc='Target variables')

    _sample_ratio=(0.8,0.2,0)
    def _get_sratio(self): return self._sample_ratio
    def _set_sratio(self,value):
        # TODO: provjeriti da li je tuple ili lista i dopuniti s nulama do dužine 3
        sr=float(sum(value))
        self._sample_ratio=tuple([r/sr for r in value])
        #self.shuffle(new_sample_ratio=value)
    sample_ratio=property(_get_sratio,_set_sratio,doc='Sample ratio (train,verify,test)')


    # lista neuralnih mreža
    nns=[]


    # private variables
    _nnsample_rnd=None # random array za određivanje "nnsample"
    _invars_null=None # boolean array, true -- at least one column is null (NAN)
    _outvars_null=None

    def _find_null_samples(self, cols):
        """ Vraća boolean array. True za sample-ove koji imaju barem jednu kolonu null (nan) """
        #cols=set(cols)
        nn=np.zeros(len(self.df),dtype=np.bool)
        for v in cols:
            nn=np.bitwise_or(nn,self.df[v].isnull())
        return nn

    def _set_nnsample(self):
        #if not (isinstance(self._invars,pandas.Series) and isinstance(self._outvars,pandas.Series)):
        if self._invars==None or self._outvars==None or self._nnsample_rnd==None:
            return
        self.df['nnsample']=1   #train
        self.df['nnsample'][self._nnsample_rnd>self.sample_ratio[0]]=2 #verify
        self.df['nnsample'][self._nnsample_rnd>sum(self.sample_ratio[0:2])]=3 #test

        # not null samples
        if isinstance(self._invars_null,pandas.Series):
            self.df['nnsample'][self._invars_null]=0
        if isinstance(self._outvars_null,pandas.Series):
            self.df['nnsample'][self._outvars_null]=0


    def shuffle(self):
        """ Promiješa sample-ove """
        #print "MIJEŠAM !!!"
        # New random array
        self._nnsample_rnd = np.random.rand(len(self.df))
        self._df['nnrand']=self._nnsample_rnd
        self._set_nnsample()


    def _set_vars(self,input_vars=None,target_vars=None):

        # TODO: Provjera da li postoje kolone
        if input_vars != None:
            if isinstance(input_vars,basestring): input_vars=[input_vars]
            self._invars=input_vars
            self._invars_null=self._find_null_samples(self._invars)
        if target_vars != None:
            if isinstance(target_vars,basestring): target_vars=[target_vars]
            self._outvars=target_vars
            self._outvars_null=self._find_null_samples(self._outvars)

        self._set_nnsample()

    def __init__(self, data, input_vars, target_vars, sample_ratio=(80,20,0)):
        """
        data -- pandas.DataFrame sa podacima
        input_vars -- list of column names with input data
        target_vars -- list of column names with target data
        sample_ratio -- tuple with relative ratio of train/verify/test samples
                     -- new column "nnsample" will be added in DataFrame with:
                         0 -- ignored (all samples with some of vars is missing)
                         1 -- train samples
                         2 -- verify samples
                         3 -- test samples
        """
        # lista neuralnih mreža
        self.nns=[]

        # TODO: Provjera da li je DataFrame
        self._df=data

        self._set_vars(input_vars,target_vars)

        if not ('nnsample' in self._df.columns):
            if sample_ratio != None: self.sample_ratio=sample_ratio
            if 'nnrand' in self._df.columns:
                self._nnsample_rnd=self._df['nnrand'].values
                self._set_nnsample()
            else:
                self.shuffle()


    def sample_index(self,sample=None):
        # TODO Ovo treba promjenit
        if sample == None: sample=(0,1,2,3)

        if isinstance(sample,int):
            sample=(sample,)

        nn=np.zeros(len(self.df),dtype=np.bool)
        nnsample=self.df["nnsample"].values
        for s in sample:
            nn=np.bitwise_or(nn,nnsample==s)
        return nn

    def input_data(self,sample=None):
        #return self.df[self._invars][self.sample_index(sample)]
        if sample==None:
            return self.df.as_matrix(self._invars)
        else:
            if type(sample) in [int,tuple]:
                ind=self.sample_index(sample)
            elif type(sample) in [list,np.ndarray]:
                ind=np.array(sample)
            return self.df.as_matrix(self._invars)[ind]

    def target_data(self,sample=None):
        #return self.df[self._outvars][self.sample_index(sample)]
        if sample==None:
            return self.df.as_matrix(self._outvars)
        else:
            if type(sample) in [int,tuple]:
                ind=self.sample_index(sample)
            elif type(sample) in [list,np.ndarray]:
                ind=np.array(sample)
            return self.df.as_matrix(self._outvars)[ind]

    def output_data(self,i,sample=None):
        net=self.nns[i]
        tst=net.test(self.input_data(sample=sample),self.target_data(sample=sample),iprint=0)
        out=tst[0]
        return out

    def residual_data(self,i,sample=None):
        rez=self.target_data(sample=sample)-self.output_data(i,sample)
        return rez

#    def output_data(self,sample=1):
#        if not isinstance(sample,tuple):
#            sample=(sample,)
#
#        nn=np.zeros(len(self.df),dtype=np.bool)
#        for s in sample:
#            nn=np.bitwise_or(nn,self.df["nnsample"]==s)
#
#        return self.df[self._invars][nn]

    def add_mlp(self,n_hidden):
        """
        Dodajem novu neuralnu mrežu s n_hidden neurona u skrivenom sloju.
        n_hidden može biti broj ili tuple
        """
        ind=len(self.nns)
        nin=len(self._invars)
        nout=len(self._outvars)
        if isinstance(n_hidden,int): n_hidden=(n_hidden,)
        arch=(nin,)+n_hidden+(nout,)
        net=ffnet(mlgraph(arch))

        self.nns.append(net)

        return ind

    def train_nn(self,i,window=10, maxiter=5000,iprint=True,method='rprop'): #, method='cg'):        
        """
        Treniranje pojedine mreže (i-te).
        window - po koliko treniram pa onda testiram verification set
        maxiter - maksimalni broj iteracija
        method - metoda ... 'rprop','cg','momentum' .....rprop ću zasad
        Treniram sve do maxiter ili kad se greška od verification pokvari
        """

        import copy
        net=self.nns[i]
        #df=self.df

        #trind=df['nnsample']==1
        #veind=df['nnsample']==2

        tr_input_data=self.input_data(1)    #df[self._invars][trind]
        tr_target_data=self.target_data(1)  #df[self._outvars][trind]
        tr_target_std = tr_target_data.std(axis=0)   # trebao bih dobiti sd za svaku varijablu

        ve_input_data=self.input_data(2)    #df[self._invars][veind]
        ve_target_data=self.target_data(2)   #df[self._outvars][veind]
        ve_target_std = ve_target_data.std(axis=0)

        if iprint:
            print "std(training)     = " , tr_target_std
            print "std(verification) = " , ve_target_std
        #tr_input_norm, tr_target_norm = net._setnorm(tr_input_data,tr_target_data)
        #ve_input_norm, ve_target_norm = net._setnorm(ve_input_data,ve_target_data)

        #net.randomweights()

        ve_sdr_old=100 # početni veliki sd ratio
        net_old=None
        iters=0; # početni window
        if method=='rprop': xmi=0.1
        while iters<maxiter:
            iters=iters+window
            #net.train_cg(tr_input_data,tr_target_data,maxiter=window,disp=False)
            if method=='rprop':
                xmi=net.train_rprop(tr_input_data,tr_target_data,maxiter=window,xmi=xmi)
            elif method=='cg':
                net.train_cg(tr_input_data,tr_target_data,maxiter=window,disp=False)
            elif method=='momentum':
                net.train_momentum(tr_input_data,tr_target_data,maxiter=window)

            tr_target_nn = net.call(tr_input_data)
            tr_error=tr_target_nn - tr_target_data
            tr_sdr = (tr_error.std(axis=0)/tr_target_std).mean()

            ve_target_nn = net.call(ve_input_data)
            ve_error = ve_target_nn - ve_target_data
            ve_sdr = (ve_error.std(axis=0)/ve_target_std).mean()

            if iprint: print "%5d: %0.3f, %0.3f" % (iters,tr_sdr,ve_sdr)
            if ve_sdr>ve_sdr_old:
                net=net_old
                iters=iters-window
                break
            ve_sdr_old=ve_sdr
            net_old=copy.copy(net)

        self.nns[i]=net
        stats=Bunch()
        tt_tr=net.test(tr_input_data,tr_target_data,iprint=0)
        stats.regression_tr=tt_tr[1];
        if iprint: print 'training:\t'+'\t'.join(['%0.3f'%t[2] for t in stats.regression_tr])

        tt_ve=net.test(ve_input_data,ve_target_data,iprint=0)
        stats.regression_ve=tt_ve[1];
        if iprint: print 'verification:\t'+'\t'.join(['%0.3f'%t[2] for t in stats.regression_ve])

        te_input_data=self.input_data(3)
        if len(te_input_data)>0:
            te_target_data=self.target_data(3)
            tt_te=net.test(te_input_data,te_target_data,iprint=0)
            stats.regression_te=tt_te[1]
            if iprint: print 'test:\t'+'\t'.join(['%0.3f'%t[2] for t in stats.regression_te])

        stats.iters=iters
        return net,stats

    def sensitivity(self,i,samples):
        """
        Sensitivity analysis
        """
        inp=self.input_data(samples)
        tar=self.target_data(samples)
        tar_std=tar.std(axis=0)

        net=self.nns[i]

        nin=inp.shape[1]
        mn=inp.mean(axis=0)

        out=net.call(inp)
        err_std=((out-tar).std(axis=0)/tar_std).sum()

        sens=np.zeros(nin)
        for iin in range(nin):
            inpm=inp
            inpm[:,iin]=mn[iin]
            out=net.call(inpm)
            errm_std=((out-tar).std(axis=0)/tar_std).sum()
            sens[iin]=errm_std/err_std

        return sens

    def test(self,i,sample,iprint=1):
        net=self.nns[i]
        inp=self.input_data(sample)
        tar=self.target_data(sample)

        out,t=net.test(inp,tar,iprint=0)
        sens=self.sensitivity(i,sample)
        s=''
        s=s+ "inp/tar\tvariable\tN\tmin\tmax\tmean\tstd\tr\tsensitivity"
        for v in range(len(self.invars)):
            a=inp[:,v]
            s=s+'\n'+ "input\t%s\t%d\t%f\t%f\t%f\t%f\t\t%f" % (self.invars[v],a.size,a.min(),a.max(),a.mean(),a.std(),sens[v])
        for v in range(len(self.outvars)):
            a=tar[:,v]
            s=s+'\n'+ "target\t%s\t%d\t%f\t%f\t%f\t%f" % (self.outvars[v],a.size,a.min(),a.max(),a.mean(),a.std())
            a=out[:,v]
            s=s+'\n'+ "output\t%s\t%d\t%f\t%f\t%f\t%f\t%f" % (self.outvars[v],a.size,a.min(),a.max(),a.mean(),a.std(),t[v][2])

        if iprint==1:
            print s
        return s

def worker_train(ens,desc):
    #print "worker_train: "+repr(desc)
    net,stats=ens.train_nn(desc.n,desc.window,iprint=False,maxiter=desc.maxiter,method=desc.method)
    #print "worker_train: treniranje gotovo: "+repr(stats)
    stats.net=net
    return stats

class TrainThreaded:

    def __init__(self,project,filename,outfolder,
                 maxiter=5000, window=50, ntimes=10,nthreads=4,counter=0,method='rprop'):
        #project = ime projekta
        #filename = ime file-a gdje ću pisati izlaze
        #outfolder = folder gdje snimam mreže
        #ens = ensemble sa podacima
        #hid = lista tupleova ili integera sa brojem neurona u layerima
        #ntimes = koliko puta treniram svaku konfiguraciju
        #nthreads = broj threadova

        #hid=[(20,20),(30,30),(40,40),(50,50)]
        #fid=safeFile(r'E:\WORK\VJETAR\Analiza\spectral_nn_3\find_net.txt')
        self.project=project
        self.outfolder=outfolder
        self.filename=filename
        self.lock=threading.Lock()
        self.counter=counter

        self.maxiter=maxiter
        self.window=window
        self.ntimes=ntimes
        self.nthreads=nthreads
        self.method=method

        self.num_in_qeue=0

    def train(self,ens,hid):

        server=pp.Server(ncpus=self.nthreads)

        for i in range(self.ntimes):
            for hh in hid:
                #self.counter=self.counter + 1

                n=ens.add_mlp(hh)
                desc=Bunch(hh=hh,n=n,window=self.window,maxiter=self.maxiter,method=self.method)
                #net,stats=ens.train_nn(n,desc.window,iprint=False,maxiter=desc.maxiter)
                self.wait_for_queu(self.nthreads) # čekam dok nebude mjesta u quue
                server.submit(worker_train,(ens,desc),(),(("bunch",)),self.write,(desc,))
                self.inc()
        server.wait()
        server.destroy()
        #self.close()

    #def close(self):
    #    self.fid.close()


    def write(self,desc,stats):

        self.lock.acquire()
        self.counter += 1
        #print "write: "+repr(stats)
        fid=open(self.filename,'a')

        r2=np.array([t[2]**2 for t in stats.regression_ve])
        print "%3d: hh=%8s, iters=%7d, r2_ve=%0.3f"%(self.counter,repr(desc.hh),stats.iters,r2.mean()),

        if 'regression_te' in stats:
            r2=np.array([t[2]**2 for t in stats.regression_te])
            print ", r2_te=%0.3f"%r2.mean(),

        print

        #print desc,stats
        #print "write: 1"
        nname="%s_%03d_%s.ffnet"%(self.project,self.counter,repr(desc.hh))
        #print "write: 2: "+os.path.join(self.outfolder,nname)
        savenet(stats.net,os.path.join(self.outfolder,nname))
        #print "write: 3"

        fid.write('%d\t%s\t%d'%
            (self.counter,'_'.join([str(h) for h in desc.hh]),len(stats.net.conec)))
        #print "write: 4"

        #tt=net.test(ens.input_data(2),ens.target_data(2),iprint=0)
        #print "1"
        fid.write('\t'+'\t'.join(['%0.3f'%rr for rr in r2]))
        #print "2"
        fid.write('\t'+'%0.3f'%r2.mean())
        fid.write('\n')
        #print "3"
        #fid.flush()
        #print "4"
        fid.close()
        #print "Gotovo zapisivanje. ..."
        self.num_in_qeue -= 1
        self.lock.release()

    def inc(self):
        self.lock.acquire()
        self.num_in_qeue +=1
        self.lock.release()

    def dec(self):
        self.lock.acquire()
        self.num_in_qeue -= 1
        self.lock.release()

    def wait_for_queu(self,num):
        while self.num_in_qeue >= num:
            time.sleep(5)





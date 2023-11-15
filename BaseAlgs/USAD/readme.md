# USAD

### **USAD : UnSupervised Anomaly Detection on Multivariate Time** Series



## Getting Started



#### Run the code

```
python main.py
```

If you want to change the default configuration, you can edit `config.yml`  or overwrite the config in `main.py` using command line args. For example:

```
python main.py --dataset=SMAP --max_epochs=20
```

```
python main.py --dataset=SMAP --restore_dir=model # restore model from `restore dir` instead of training it
```



Naming convention

For network parameters of  trained model and resulting scores of train data and test data, the directory including them follows the naming convention that dataset name + '_' + max_epochs. For example:

`/USAD/model/MSL_20/`

which means this directory stores the nn parameters of the model trained under `MSL` dataset and 20 epochs.

 





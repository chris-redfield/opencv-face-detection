#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
from PIL import Image


# In[2]:


print("loading train dataset")
path = Path('data/')
#path.ls()


# In[3]:


def is_obama(x): 
    return 'obama' in x


# In[4]:


dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42, bs=2,
    label_func=is_obama, item_tfms=Resize(224))


# In[5]:


print("training model")
learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fit(4)


# In[6]:


learn.fine_tune(1)


# In[7]:


tst_image = "data/obama-1.jpg"

uploader = SimpleNamespace(data = [tst_image])
img = PILImage.create(uploader.data[0])


# In[8]:


img


# In[9]:


is_obama_,_,probs = learn.predict(img)


# In[10]:


is_obama_


# In[11]:


print("Testing image", tst_image)


# In[12]:


print(f"Probability it's obama: {probs[1].item():.6f}")


# In[13]:


print("model exported to models/export.pkl")
path = Path('models/')
learn.path = path
learn.export()


# In[14]:


#learn_inf = load_learner('data2/export.pkl')


# In[15]:


#learn_inf.predict('data/obama-1.jpg')


# In[ ]:





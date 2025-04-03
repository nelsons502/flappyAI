#!/usr/bin/env python
# coding: utf-8

# # Flappy Bird AI
# This is the code I will use to build a Flappy Bird AI.

# In[1]:


import torch
print(f"Is MPS available? {torch.backends.mps.is_available()}")


# In[2]:


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tensor = torch.randn(10, 10).to(device)
print(tensor)


# In[ ]:





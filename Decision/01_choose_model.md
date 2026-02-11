## Model Selection
How to choose which model to use, interms of paramtere size

- Dateset size relative to model size
    - you wan 2k sample for per billion model size
    - if we have 15k samples, then we can choose upto 8B model parameter
    - Too large model with less data == overfitting
-  Find out capacity of existing model
  
### Model
- Qwen3-8
  
#### Dataset
- We are using `shaneperry0101/health-chatbot`
- it will have a prompt and response

### Metrics 
-  So the metrics we are looking here based on our medical dataset is: 
   -  Factual accuracy/ Correctness
   -  Relevance
   -  Safety


Pre-Evaluate

medicalinfo =60/10
gk = 90/10

train ->eval again---> 70/10
gk 50/10

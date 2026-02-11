### LLM Evaluation : 

    - What is prompt engineering
         prompt engineering is practice of crafing and optimizing prompt get the best possible outcome form an AI model(LLM)

    - what is finetuning 
        further training a pretrained model on a specific dataset
        - when we do finetuning?
            1.When we need domain specific knowlege in llm.
            2. When prompt engineering doesnt give you optimial result
            3. We need a spefic style or tone in llm response
        - when we don't need finetuning:
            1. when prompt engineering delivers optimal result
            2. when you have a little training data (less than 1000 sample)
            3. The data changes frequently (use rag)


#### FineTuning

 - whta we need:
    1. model (llm) - SLM (0 - 8B)
    2. Dataset 
    3. GPU, Training lib (Huggingface, axotl, unsloth, llamafactory, CollosalAI)
- Prerequsite
    1. Eval the current model based on the custom dataset


### Evaluation
   1. Benchmarks - Its a larage dataset on sepcific domain.  It will have a question , ref answer and metric (accuracy-multiplre choice question)

   2. Metrics
   
    No, not all benchmarks use accuracy. The metric depends on the type of task.
    Multiple choice tasks → Accuracy

    MedQA, MMLU, ARC
    Simple: right or wrong

    Text generation tasks → ROUGE, BLEU, BERTScore

    Summarization, translation
    No single "correct" answer, so you measure similarity to a reference

    Code generation → pass@k

    HumanEval
    Does the code actually run and pass test cases?

    Open-ended conversation → Human eval / LLM-as-Judge

    No single correct answer
    Scored on helpfulness, coherence, relevance

    Classification → F1, Precision, Recall

    Sentiment analysis, spam detection
    Handles imbalanced data better than accuracy
    



               





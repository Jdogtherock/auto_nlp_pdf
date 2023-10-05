# **9/6/23**
- [x] implement security that prevents user from uploading a pdf that doesnt contain any actual words. So if the pdf only contains images, whitespace, stopwords, or punctuation, prompt to try again.
- [ ] plan method of summarization (embeddings, LLM, etc.)

# **9/7/23**:
- [ ] add top 10 bi-grams and tri-grams to the pdf eda (non stopwords)
- [ ] choose embedding method,
- [ ] summarization llm
- [ ] keyword extraction llm
- [ ] zero-shot classification model
- [ ] NER llm
- [ ] q&a llm

# **9/10/23**:
- implemented zero shot classification and unit testing
- need to further optimize accuracy (and possibly speed)
- TODO: Optimize Classification Accuracy & Speed: going to try blending the text chunks, once the app is deployed. might improve accuracy by holding context
- tried speedup by setting device=0 (pipeline runs on GPU)
- chat gpt speedup suggestions:
Yes, you can utilize GPU clusters to speed up computations when working with Hugging Face's pipeline. However, there are a few things to note:

Single-GPU Usage: The pipeline in Hugging Face's Transformers library natively supports GPU acceleration. If you have a CUDA-compatible GPU available on your machine, you can simply move the model to the GPU by setting the device argument to the desired GPU ID (0 for the first GPU, 1 for the second, etc.):

python
Copy code
from transformers import pipeline

nlp = pipeline("sentiment-analysis", device=0)  # to run on the first GPU
Multi-GPU & Distributed Inference: If you want to run inference on a cluster of GPUs, things get more involved. The pipeline itself doesn't have built-in support for distributed inference over multiple GPUs or machines.

For multi-GPU setups on a single machine, you can use DataParallel from PyTorch to parallelize computations over all available GPUs.
For distributed setups across multiple machines, you'd need to look into utilities like DistributedDataParallel from PyTorch.
In both cases, you'd need to set up and manage the distribution of data across the GPUs yourself, which means you'd have to move away from the simplicity of the pipeline abstraction.

Model Parallelism: Some models are too large to fit in the memory of a single GPU. Libraries like DeepSpeed and Megatron-LM provide techniques for model parallelism, where different parts of a model run on different GPUs. This allows training and deployment of models that wouldn't otherwise fit in memory. However, integrating these with Hugging Face's pipeline would require significant custom code.

Cloud GPU Clusters: Cloud providers like AWS, GCP, and Azure offer GPU clusters that you can provision. Once set up, you can deploy your model there and potentially use technologies like NVIDIA's NCCL for multi-GPU communication and orchestration tools like Kubernetes to manage the cluster.

Hugging Face Accelerated Inference: Hugging Face offers an accelerated inference API which you might consider if you're okay with sending data to their servers. This provides you with faster inference times without the hassle of managing the infrastructure yourself.
- once my transformers are implemented, focus on speedup, accuracy, and trying to compare fine tuned models versus the general base ones on the first deployment.

# **9/11/23**:
- implementing the summarization method
- RULE OF THUMB: the more intermediates there is, the slower the process will be
- ex. calling the huggingface pipeline instead of using a direct API will be slower, since the pipeline calls the API. So itd be us -> pipeline -> api instead of us -> api
- definitely need to perform speed & accuracy optimizations once I get the first iteration deployed
- Zero Shot: decent accuracy, but i want to see if it can compute faster
- Summarize: decent accuracy, but same in terms of faster computation. Also, I noticed small syntactic errors in the summary. (ex. , instead of . and '98th century). Need to figure out ways to fix this.
- Improving accuracy: model that performed better on benchmark, larger model
- Improving speed: direct api usage, checking for redundant computation, distributed computing


# **9/16/23**:
- going to utilize the model's token input length in order to chunk, not custom chunks
- for classification, going to use overlapping chunks to retain context
- for summarization, going to use discrete chunks so information isnt repeated
- adding a models markdown page with the models used and a short description

# **9/17/23**:
- fully implemented zero-shot
- trying to fix summarize, a lot of repeating words and phrases (need to lookup how to fix)
- messing with abstractive vs. extractive summarization (for speed & performance curiousity)
- going to mess around later, moving onto NER
- NER is very messed up, need to fix

# **9/18/23**:
- going to split csv autoML and pdf autoML into two different streamlit applications, just easier that way
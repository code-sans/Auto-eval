# Auto-eval
The education sector is being transformed by artificial intelligence through the customization, 
efficiency and accessibility of learning. Recent advancements in artificial intelligence have sparked significant 
interest in automating the evaluation of handwritten answers. Traditional handwritten evaluation techniques are 
influenced by the evaluator's mental and physical state, environmental factors, human bias, emotional swings, and 
logistical challenges like storage and retrieval. Existing AI evaluation techniques, such as sequence-to-sequence 
neural networks, have shown promise but are limited by their dependence on high-performance hardware like 
GPUs, long training times, and difficulties in handling diverse scenarios. The limitations of existing NLP methods 
such as Bag of Words, TF-IDF, and Word2Vec have been overcome by the state-of-the-art method called 
Bidirectional Encoders Representation from Transformer (BERT). But BERT depends on surface-level keyword 
similarity, if the keywords are different then the accuracy is not perfect. This study presents a technique that 
combines optical character recognition (OCR) technology with DeepSeek-R1 1.5B model to create a robust, 
efficient, and accurate grading system. To overcome the above-mentioned challenges, we incorporate the Google 
Cloud Vision API to extract and convert handwritten responses into machine-readable text, providing a preprocessed input for further evaluation in this study. To check the performance of the proposed DeepSeek 
evaluation method, we compare its results with cosine similarity metrics. The finding of this technique is shown 
that proposed technique is reliable and accurate. The main aim of this study is to develop a scalable, automated, 
and effective system for grading handwritten responses by combining DeepSeek for response evaluation with the 
Google Cloud Vision API for text extraction

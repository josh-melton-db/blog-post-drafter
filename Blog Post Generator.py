# Databricks notebook source
# MAGIC %pip install dspy-ai --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() + '/serving-endpoints'

# Set up the LM
lm = dspy.Databricks(model='databricks-meta-llama-3-70b-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=1000)
dspy.settings.configure(lm=lm)

# COMMAND ----------

golden_dataset_outlines = (
    spark.read.option('header', 'true').option('multiline', 'true')
    .csv('/Volumes/josh_melton/learning/dspy_cache/blog_post_outlines.csv')
    .drop('Blog Link')
).toPandas()
synthetic_examples = spark.read.table('josh_melton.learning.dspy_examples').toPandas()
outline_trainset = [dspy.Example(abstract=row['Abstract'], outline=row['Outline']).with_inputs('abstract') for i, row in golden_dataset_outlines.iterrows()]
outline_testset = [dspy.Example(abstract=row['thoughts'][16:-73], outline=row['outlines'][16:-73]).with_inputs('abstract') for i, row in synthetic_examples.iterrows()]

# COMMAND ----------

class AbstractToOutline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("abstract -> outline")
    
    def forward(self, abstract):
        return self.prog(abstract=abstract)

# COMMAND ----------

class Assess(dspy.Signature):
    """Assess the quality of an outline along the specified dimension."""
    text_to_assess = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# COMMAND ----------

def outline_metric(gold, pred, trace=None):
    abstract, outline, output = gold.abstract, gold.outline, pred.outline
    engaging = "Does the outline define an engaging or novel topic for a Databricks blog?"
    structured = "Does the outline follow a highly structured format similar to 1a, b, c, 2a, b, c, etc?"
    introduction = "Does the outline start with a clear introduction section with a problem statement and proposed solution?"
    support = "Are there supporting sections that can be used to provide examples and support for the proposed solution to the problem statement?"
    conclusion = "Is there a clear conclusion section that can be used to emphasize what the reader has learned?"
    evals =  [dspy.Predict(Assess)(text_to_assess=output, assessment_question=question) 
              for question in [engaging, structured, introduction, support, conclusion]]
    score = sum(['yes' in e.assessment_answer.lower() for e in evals])
    return score

# COMMAND ----------

from dspy.evaluate import Evaluate

# Evaluate the baseline model
evaluate = Evaluate(devset=outline_testset[:5], metric=outline_metric, num_threads=4, display_progress=False, display_table=0)
baseline_outline_results = evaluate(AbstractToOutline())
print(baseline_outline_results)

# COMMAND ----------

from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 2-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=2, max_labeled_demos=2)

# Optimize! In general, the metric is going to tell the optimizer how well it's doing.
optimizer = BootstrapFewShot(metric=outline_metric, **config)
optimized_outliner = optimizer.compile(AbstractToOutline(), trainset=outline_trainset)

# COMMAND ----------

# Evaluate the optimized model
evaluate = Evaluate(devset=outline_testset[:5], metric=outline_metric, num_threads=4, display_progress=False, display_table=0)
optimized_results = evaluate(optimized_outliner)
print(optimized_results)

# COMMAND ----------

# lm.inspect_history(n=2)
improvement = (optimized_results / baseline_outline_results) - 1
print(f"% improvement: {improvement * 100}")

# COMMAND ----------

class SectionToParagraph(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("section, topic -> paragraph")
    
    def forward(self, section, topic):
        return self.prog(section=section, topic=topic)

# COMMAND ----------

golden_dataset_paragraphs = (
    spark.read.option('header', 'true').option('multiline', 'true')
    .csv('/Volumes/josh_melton/learning/dspy_cache/blog_post_paragraphs.csv')
).toPandas()
paragraph_trainset = [dspy.Example(section=row['Section'], topic=row['Topic'], paragraph=row['Paragraph']).with_inputs('section', 'topic') 
                      for i, row in golden_dataset_paragraphs.iterrows()]
paragraph_testset = paragraph_trainset
# TODO: generate sections from the example outlines and add them as a testset

# COMMAND ----------

def paragraph_metric(gold, pred, trace=None):
    section, paragraph, topic, output = gold.section, gold.topic, gold.paragraph, pred.paragraph
    clarity = "Is the paragraph clear, concise, and does it have continuity?"
    intention = "Does the paragraph have a clear intention and showcase how Databricks or data more generally solves some problem?"
    support = "If a supporting paragraph (not the introduction or conclusion), does the paragraph have a placeholder similar to <CODE_EXAMPLE> to provide support to its point?"
    detailed = "Does the paragraph provide excellent detail about the overall point, rather than being generic or repetitive?"
    aligned = f"Is the paragraph aligned to the target topic, {topic}?"
    evals =  [dspy.Predict(Assess)(text_to_assess=output, assessment_question=question) 
              for question in [clarity, intention, support, detailed, aligned]]
    score = sum(['yes' in e.assessment_answer.lower() for e in evals])
    if 'yes' in evals[-1].assessment_answer.lower(): score += 3
    return score

# COMMAND ----------

evaluate = Evaluate(devset=paragraph_testset[:5], metric=paragraph_metric, num_threads=4, display_progress=False, display_table=0)
baseline_paragraph_results = evaluate(SectionToParagraph())
print(baseline_paragraph_results)

# COMMAND ----------

config = dict(max_bootstrapped_demos=3, max_labeled_demos=3)
optimizer = BootstrapFewShot(metric=paragraph_metric, **config)
optimized_section_writer = optimizer.compile(SectionToParagraph(), trainset=paragraph_trainset)

# COMMAND ----------

evaluate = Evaluate(devset=paragraph_testset[:5], metric=paragraph_metric, num_threads=4, display_progress=False, display_table=0)
optimized_paragraph_results = evaluate(optimized_section_writer)
print(optimized_paragraph_results)

# COMMAND ----------

improvement = (optimized_paragraph_results / baseline_paragraph_results) - 1
print(f"% improvement: {improvement * 100}")

# COMMAND ----------

dspy_topic = 'Using Databricks and DSPy to create AI Systems'
dspy_abstract = '''
In this blog post, we'll outline our approach to taking a collection of thoughts about some customer problem and turning them into the draft of a blog post. We propose using tools like Databricks Foundation Models, DSPy, and datasets from annotations of previous blog posts to create an automated system which can produce rough drafts of blog posts in a matter of seconds. We hope a blog on this internal facing use case can provide helpful insights for customers aiming to provide value to their business using the tools mentioned. DSPy eliminates brittle, model-specific prompt engineering tasks and turns them into optimized language model systems. Foundation models are an extremely simple and quick way to access the latest open source large language models. Datasets formed using of previous blog posts can be used to fine tune the system. The focus should not be on the benefit to the Databricks field, it should be on which insights gleaned from this project that customers can leverage in their own pipelines to create more robust, production ready AI systems - insights such as how DSPy + foundation models are very simple to use for their own end to end AI processes, or how Databricks enables using unstructured data to build those processes on their data.
'''
dspy_outline_output = optimized_outliner(dspy_abstract.strip()).outline

import re
def parse_outline(outline):
    output = re.split(r'\n|\d+\.', outline)
    return [line.strip() for line in output if line.strip()]

outline_sections = parse_outline(dspy_outline_output)
dspy_paragraph_output = [optimized_section_writer(section, dspy_topic) for section in outline_sections]
print([output.paragraph for output in dspy_paragraph_output])

# COMMAND ----------

mlflow_topic = 'Using MLflow to predict the probability of multiple classes'
mlflow_abstract = '''
Databricks users often require extensions to the standard MLflow interfaces, such as logging artifacts alongside a model or altering the way it makes predictions. In this blog post, we'll describe an approach to using a custom MLflow model to predict the probability of multiple classes. First, we'll train a simple classification model on a dummy dataset. Next, we'll create a custom MLflow PyFunc model to wrap the initial model and return the top N classes and their probabilities. Finally, we'll deploy the model using MLflow and leverage it for inference.
'''
mlflow_outline_output = optimized_outliner(mlflow_abstract.strip()).outline

import re
def parse_outline(outline):
    output = re.split(r'\n|\d+\.', outline)
    return [line.strip() for line in output if line.strip()]

outline_sections = parse_outline(mlflow_outline_output)
mlflow_paragraph_output = [optimized_section_writer(section, mlflow_topic) for section in outline_sections]
print(mlflow_outline_output)
print("\n\n")
print([output.paragraph for output in mlflow_paragraph_output])

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC import mlflow

# COMMAND ----------

outliner_save_path = '/Volumes/josh_melton/blogs/dspy_models/outliner.json'
optimized_outliner.save(outliner_save_path)
section_writer_save_path = '/Volumes/josh_melton/blogs/dspy_models/section_writer.json'
optimized_section_writer.save(section_writer_save_path)

# COMMAND ----------

import os
os.environ['token'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

class DSPyWrapper(mlflow.pyfunc.PythonModel):
    # define steps to initialize model
    def load_context(self, context):
        outliner = AbstractToOutline()
        outliner.load(path=outliner_save_path)
        self.outliner = outliner
        section_writer = SectionToParagraph()
        section_writer.load(path=section_writer_save_path)
        self.section_writer = section_writer
        self.dspy_setup()

    def dspy_setup(self):
        import dspy
        import os
        token = os.environ['token']
        url = 'https://oregon.cloud.databricks.com/serving-endpoints'
        lm = dspy.Databricks(model='databricks-meta-llama-3-70b-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=1000)
        dspy.settings.configure(lm=lm)

    def parse_outline(self, outline):
        import re
        output = re.split(r'\n|\d+\.', outline)
        return [line.strip() for line in output if line.strip()]
    
    def draft_blog(self, row):
        topic, abstract = row['topic'], row['abstract']
        outline_output = self.outliner(abstract.strip()).outline
        outline_sections = self.parse_outline(outline_output)
        paragraph_output = [self.section_writer(section, topic).paragraph for section in outline_sections]
        return pd.Series([outline_output, paragraph_output])

    def predict(self, context, input_df):
        output = input_df.apply(self.draft_blog, axis=1, result_type='expand')
        output.columns = ['outline', 'paragraphs']
        return output

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path="model", python_model=DSPyWrapper())

# COMMAND ----------

import pandas as pd

model_uri = f'runs:/{run.info.run_id}/model'
loaded_model = mlflow.pyfunc.load_model(model_uri)
input_data = pd.DataFrame({
    'topic': [dspy_topic, mlflow_topic],
    'abstract': [dspy_abstract, mlflow_abstract],
})
response = loaded_model.predict(input_data)
display(response)

# COMMAND ----------

from mlflow.models import infer_signature
signature = infer_signature(input_data, response)
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path="model", python_model=DSPyWrapper(), signature=signature, input_example=input_data)

model_uri = f'runs:/{run.info.run_id}/model'
loaded_model = mlflow.pyfunc.load_model(model_uri)
input_data = pd.DataFrame({
    'topic': [dspy_topic, mlflow_topic],
    'abstract': [dspy_abstract, mlflow_abstract],
})
response = loaded_model.predict(input_data)
display(response)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_name = 'josh_melton.blogs.blog_post_drafter'
result = mlflow.register_model(model_uri, name=model_name)

# COMMAND ----------



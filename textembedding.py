from google.cloud import aiplatform
from google.cloud import aiplatform
from google.oauth2 import service_account
aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project='motion-generator-386613',

    # the Vertex AI region you will use
    # defaults to us-central1
    location='us-central1',

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://my_staging_bucket',

    # custom google.auth.credentials.Credentials
    # environment default creds used if not set
    # credentials=my_credentials,

    # customer managed encryption key resource name
    # will be applied to all Vertex AI resources if set
    # encryption_spec_key_name=my_encryption_key_name,

    # the name of the experiment to use to track
    # logged metrics and parameters
    experiment='my-experiment',

    # description of the experiment above
    experiment_description='my experiment decsription',
    credentials=service_account.Credentials.from_service_account_file('credentials.json')
)

from vertexai.preview.language_models import TextEmbeddingModel


def text_embedding(text_input):
  """Text embedding with a Large Language Model."""
  model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
  embeddings = model.get_embeddings(text_input)
  for embedding in embeddings:
      vector = embedding.values
      print(f'Length of Embedding Vector: {len(vector)}')
      print(vector)
      return vector
  
text_input_list = ['hello,how are you']
vector = text_embedding(text_input_list)
print("text_embedding::::",len(vector))
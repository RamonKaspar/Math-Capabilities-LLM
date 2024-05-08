from techniques.Baseline import Baseline
from techniques.PaL import PaL

method = PaL(name="Baseline", dataset="arithmetic", service="openai", model="gpt-3.5-turbo", temperature=0.1, max_token=200)

res = method.query_with_detailed_response("What is 3+3?")
print(res)



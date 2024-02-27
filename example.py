import os
os.environ["OPENAI_API_KEY"] = "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"


# Cinderella story defined in sample.txt
with open('Demo/sample.txt', 'r') as file:
    text = file.read()

print(text[:100])

from raptor import RetrievalAugmentation

RA = RetrievalAugmentation()

# construct the tree
RA.add_documents(text)


question = "How did Cinderella reach her happy ending"

answer = RA.answer_question(question=question)

print("Answer: ", answer)
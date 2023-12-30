import gradio as gr

from transformers import pipeline
from spacy import displacy



ner_pipeline = pipeline("ner")


def ner(text):
    ner_results =  ner_pipeline(text, aggregation_strategy="average")
    ents = [
            {
                'start': ent['start'], 
                'end': ent['end'],
                'label': ent['entity_group']
            }
            for ent in ner_results
        ]
    spacy_format = {
        'text': text,
        'ents': ents,
    }
    html = displacy.render(spacy_format, style="ent", manual=True, page=True)
    return html


demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
    # Ner
    Start typing below to see the output.
    """
    )
    input = gr.Textbox(placeholder="Enter text ...")
    output = gr.HTML()
    examples=['dsds']

    input.change(fn=ner, inputs=input, outputs=output)

# demo = gr.Interface(
#     fn=ner,
#     inputs='text',
#     outputs='html',
#     live=True,
#     examples=['My name is Eyal'],
#     # layout='vertical',
# )

demo.launch(share=False)
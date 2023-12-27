import gradio as gr

from transformers import pipeline
from spacy import displacy



ner_pipeline = pipeline("ner")


def flip_text(text):
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
    # Flip Text!
    Start typing below to see the output.
    """
    )
    input = gr.Textbox(placeholder="Flip this text")
    output = gr.HTML()

    input.change(fn=flip_text, inputs=input, outputs=output)

demo.launch(share=True)
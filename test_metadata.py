from chatbot import process_document
import json

doc = process_document('training_data/simulations/pml_xmax.in', 'simulation')
if doc:
    print("METADATA:")
    print(json.dumps(doc[0]['metadata']['simulation_params'], indent=2))
    print("\nDESCRIPTION:")
    print(doc[0]['page_content']) 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("Swamitucats/M2M100_Sanskrit_English")
tokenizer = AutoTokenizer.from_pretrained("Swamitucats/M2M100_Sanskrit_English")

sanskrit_text = "भूमण्डले इतरदेरोभ्यः पूर्वमेव भारतवर्षे चिकित्साशाख्रमासीत्‌ समुन्नतिपथारूढमितीतिहासविदां मतम्‌"
inputs = tokenizer(sanskrit_text, return_tensors="pt")
outputs = model.generate(**inputs)
english_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(english_translation)
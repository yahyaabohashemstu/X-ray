from transformers import GPTJForCausalLM, GPT2Tokenizer
import torch

# تحميل النموذج والمحسن (tokenizer)
model_name = "EleutherAI/gpt-j-6B"  # يمكنك استخدام "EleutherAI/gpt-neo-2.7B" إذا كنت ترغب في نموذج أصغر
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# جعل النموذج في وضع التقييم
model.eval()

# دالة لتوليد الردود
def generate_response(prompt):
    # تحويل النص إلى رموز (tokens)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # توليد النص من النموذج
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
    # تحويل الرموز إلى نص
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# بدء المحادثة
print("أهلاً! أنا بوت المحادثة الخاص بك. (اكتب 'وداعاً' للخروج)")

while True:
    user_input = input("أنت: ")
    
    if user_input.lower() == "وداعاً":
        print("البوت: إلى اللقاء!")
        break
    
    # الحصول على الرد من النموذج
    bot_response = generate_response(user_input)
    
    print("البوت: " + bot_response)

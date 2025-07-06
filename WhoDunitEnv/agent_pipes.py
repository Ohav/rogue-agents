import openai
import consts

device = 'cuda'

def gemma_processing(message):
    new_messages = []
    last_role = ''
    for i in range(len(message)):
        if message[i]['role'] == 'system':
            message[i]['role'] = 'user'
        if last_role == message[i]['role']:
            new_messages[-1]['content'] = new_messages[-1]['content'] + '\n' + message[i]['content']
        else: 
            new_messages.append(message[i])
        last_role = message[i]['role']
    return new_messages

class TransformersPipe:
    def __init__(self, model, **kwargs):
        print(f"loading pipeline {model}")
        self.is_fast_chat = False
        self.is_open_ai = False
        self.tokenizer = None
        self.seed = 42
        self.gpu_count = kwargs.get('gpu_count', 0)
        
        self.save_logprobs = False
        self.last_logprobs = None
        
        print(f"Model GPUs: {self.gpu_count}")
        self.url = kwargs.get('url', '')
        self.input_processing = lambda v: v
        if (self.gpu_count != 0) and (model == 'gemma' or 'mistral' in model):
            # Pipeline
            model = consts.MODEL_TO_FULL_PATH.get(model, model)
            self.is_fast_chat = True
        elif model.startswith('gpt'):
            self.is_open_ai = True
        else:
            openai.base_url = self.url
            print(f"Model URL: {openai.base_url}")

            model = consts.MODEL_TO_FULL_PATH.get(model, model)
            is_local = (self.gpu_count > 0 or not self.url)
            self.is_fast_chat = is_local
            self.is_open_ai = not is_local # Use openai format
            openai.api_key = "EMPTY"

        
        if not self.is_open_ai:
            from transformers import pipeline
            self.pipe = pipeline("text-generation", model=model, device_map='auto', do_sample=True)
        elif self.is_open_ai:
            self.pipe = openai
            self.model = model

        if 'gemma' in self.model:
            self.input_processing = gemma_processing

    def set_save_logprobs(self, val):
        if not self.is_open_ai:
            raise Exception("Logprob saving not implemented with non-openai frameworks")
        self.save_logprobs = val
        if not self.save_logprobs:
            self.last_logprobs = None

    def get_last_logprobs(self):
        if self.save_logprobs:
            return self.last_logprobs
        return None

    def can_batch(self):
        return self.is_fast_chat

    def __call__(self, message, temperature):
        if self.is_open_ai:
            return self.run_open_ai(message, temperature)
        if self.is_fast_chat:
            return self.run_fast_chat(message, temperature)
        return self.run_batch_transformers([message], temperature)[0]
    
    def run_pipeline(self, message, temperature):
        res = self.pipe(message, max_new_tokens=512, pad_token_id=50256, temperature=temperature)
        return res[-1]['generated_text'][-1]['content']

    def run_open_ai(self, message, temperature):
        message = self.input_processing(message)
        completion = self.pipe.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=temperature,
            logprobs=self.save_logprobs,
            top_logprobs=10 if self.save_logprobs else None
        )
        if self.save_logprobs:
            self.last_logprobs = completion.choices[0].logprobs.content
            self.last_logprobs = [[(p.token, p.logprob) for p in gen.top_logprobs] for gen in self.last_logprobs]
        return completion.choices[0].message.content
    
    def run_fast_chat(self, message, temperature):
        return self.run_fast_chat_batch([message], temperature)[0]
    
    def run_fast_chat_batch(self, messages, temperature):
        import torch
        prompts = []
        for message in messages:
            prompt = self.tokenizer.apply_chat_template(message, return_tensors='pt').to(device)
            prompts.append(prompt)
        output_ids = self.pipe.generate(
            torch.concatenate(prompts),
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1,
            max_new_tokens=2000,
        )
        answers = []
        for i in range(len(output_ids)):
            output = output_ids[i][len(prompts[i][0]):]
            answer = self.tokenizer.decode(
                output, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            answers.append(answer)
        return answers

    def run_batch_transformers(self, message, temperature):
        tokenized = self.tokenizer.apply_chat_template(message, return_tensors='pt')
        tokenized = tokenized.to(device)
        gen_tokens = self.model.generate(tokenized, max_new_tokens=2048, do_sample=True, temperature=temperature)
        decoded = self.tokenizer.batch_decode(gen_tokens[:, tokenized.shape[1]:], skip_special_characters=True)
        return decoded

    def run_batch(self, messages, temperature):
        if self.is_fast_chat:
            return self.run_fast_chat_batch(messages, temperature)
        if self.is_open_ai:
            answers = []
            for message in messages:
                answer = self.run_open_ai(message, temperature)
                answers.append(answer)
            return answers
        
        # Transformers Pipeline
        return self.run_batch_transformers(message, temperature)

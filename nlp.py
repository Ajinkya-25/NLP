import tensorflow as tf
import numpy as np

tokenizer=tf.keras.preprocessing.text.Tokenizer(oov_token="<oov>")
max_length=40
# need mor good conversation
sentences = [
    "Hey! Whats up? Need any help?",
    "Yeah, I was just wondering what you guys offer.",
    "Oh, weve got some cool stuff! Want to know the details?",
    "Sure! Weve got features that make everything super easy.",
    "Sounds great! Whats in it for me?",
    "You get convenience, efficiency, and top-notch support.",
    "Nice! How do I get started?",
    "Just sign up on our site and follow the easy steps.",
    "Do you guys help out if I get stuck?",
    "Absolutely! Were here 24/7 for anything you need.",
    "Can I try it out before I commit?",
    "Definitely, we have a free trial so you can check it out first.",
    "What payment options do you offer?",
    "We take credit cards, debit cards, and online payments.",
    "How do I get in touch if I need help?",
    "You can reach us via email, phone, or live chat on our site.",
    "What if I forget my password?",
    "No worries, you can reset it easily through your email.",
    "Any special discounts going on?",
    "Weve got occasional deals. Check out our site or sign up for updates.",
    "Can I switch my plan later if I need to?",
    "Yep, you can upgrade or downgrade anytime through your account settings.",
    "What about refunds?",
    "We have a refund policy for paid plans. Check our terms for details.",
    "How safe is this thing?",
    "Were serious about security and use encryption to keep your info safe.",
    "Can I cancel whenever I want?",
    "Yep, you can cancel anytime through your account settings.",
    "Whats your company all about?",
    "Were all about making your life easier with great solutions.",
    "How often do you update stuff?",
    "We keep things fresh with regular updates and improvements.",
    "Can I link this with other tools?",
    "For sure! We offer integrations with various tools.",
    "Do you have an app?",
    "Yeah, weve got apps for both iOS and Android.",
    "What do I need to use your service?",
    "Just a modern web browser and a stable internet connection.",
    "How can I give feedback?",
    "Drop us a note through our feedback form on the site or contact support.",
    "Do you have a referral program?",
    "Yep, refer friends and earn rewards!",
    "How do you handle data privacy?",
    "We take your privacy seriously and follow strict data protection rules.",
    "How can I reach your sales team?",
    "Email, phone, or fill out the contact form on our site.",
    "Do you offer any training or guides?",
    "Absolutely, weve got training materials and tutorials to help you out.",
    "Can I tweak the features to my liking?",
    "Sure thing! We offer customization to fit your needs."
    "Hey kiddo, how was school today?",
    "It was okay. We had a big math test.",
    "Oh, how did it go?",
    "I think I did pretty well. I was a bit nervous though.",
    "That's great to hear! Want to grab a treat later?",
    "Sure, that sounds awesome! Maybe ice cream?",
    "Ice cream it is! What flavor do you want?",
    "Hmm, I think I'll go with chocolate chip cookie dough.",
    "Yum, good choice! By the way, did you finish your homework?",
    "Almost done. I just have a little bit left.",
    "Okay, just make sure to finish it before dinner.",
    "Got it, Mom. Can I play some video games after?",
    "Sure, but only after your homework is done. Deal?",
    "Deal! Thanks, Dad.",
    "You're welcome! Also, remember we have that family movie night tonight.",
    "Oh yeah! I almost forgot. What movie are we watching?",
    "We haven't decided yet. Maybe we can pick together later.",
    "Sounds good to me! Can I invite a friend over for a bit?",
    "As long as its okay with their parents and they dont stay too late.",
    "Got it. Ill ask them and let you know.",
    "Perfect! Lets finish up here and then well head out for that ice cream.",
    "Awesome! Cant wait. Thanks for the treat, Mom and Dad.",
    "Anytime! Were proud of you. Keep up the good work!",
    "Thanks! Ill try my best.",
    "Thats our kiddo! Now, lets get ready to go."]

tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
sequence=tokenizer.texts_to_sequences(sentences)
word_size=len(word_index)

def make_data(tokens):
    inputs=[]
    targets=[]
    for word in tokens:
        for i in range(len(word)-1):
            inputs.append(word[:i+1])
            targets.append(word[i+1])

    return tf.keras.preprocessing.sequence.pad_sequences(inputs,maxlen=max_length,padding='pre'),np.array(targets)


def shuffle():
    indices=np.arange(inputs.shape[0])
    np.random.shuffle(indices)

    shuffle_targets=targets[indices]
    shuffle_inputs=inputs[indices]
    return shuffle_inputs,shuffle_targets


inputs,targets=make_data(sequence)
inputs,targets=shuffle()

print(inputs)
print(targets)

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=word_size+1 , output_dim=128, input_length=max_length),  
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(39),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(word_size + 1, activation='softmax')
])
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'],optimizer='adam')
print(inputs.shape)
print(targets.shape)
model.fit(inputs,targets,epochs=200)

def response_generator(user_ip):
    seed=user_ip
    op=''
    while True:
        user_sequence=tokenizer.texts_to_sequences([seed])
        padded_sequence=tf.keras.preprocessing.sequence.pad_sequences(user_sequence, maxlen=max_length, padding='pre')
        ans=model.predict(padded_sequence)
        predicted_word_index = np.argmax(ans)
        print(predicted_word_index)
        predicted_word = tokenizer.index_word[predicted_word_index]
        if len(op)>50 :
            break
        seed+=" "+predicted_word
        op+=" "+predicted_word
    return op

print("welcome to the chatbot")
print(tokenizer.index_word)
while True:
    ip=input("you: ")
    ans=response_generator(ip)
    print("bot: ",ans," ")
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:02:36 2020

@author: Smruti
"""
import nltk
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """The boy's name was Santiago. Dusk was falling as the boy arrived with his herd at an abandoned church. The roof had fallen in long ago, and an enormous sycamore had grown on the spot where the sacristy had once stood. He decided to spend the night there. He saw to it that all the sheep entered through the ruined gate, and then laid some planks across it to prevent the flock from wandering away during the night. There were no wolves in the region, but once an animal had strayed during the night, and the boy had had to
spend the entire next day searching for it. He swept the floor with his jacket and lay down, using the book he had just finished reading as a pillow. He told himself that he would have to start reading thicker books: they lasted longer, and made more comfortable pillows. It was still dark when he awoke, and, looking up, he could see the stars through the half-destroyed roof. I wanted to sleep a little longer, he thought. He had had the same dream that night as a week ago, and once again he had awakened before it ended.
He arose and, taking up his crook, began to awaken the sheep that still slept. He had noticed that, as soon as he awoke, most of his animals also began to stir. It was as if some mysterious energy bound his life to that of the sheep, with whom he had spent the past two years, leading them through the countryside in search of food and water. "They are so used to me that they know my schedule," he muttered. Thinking about that for a moment, he realized that it could be the other way around: that it was he who had become accustomed to their schedule. But there were certain of them who took a bit longer to awaken. The boy
prodded them, one by one, with his crook, calling each by name. He had always believed that the sheep were able to understand what he said. So there were times when he read them parts of his books that had made an impression on him, or when he would tell them of the loneliness or the happiness of a shepherd in the fields. Sometimes he would comment to them on the things he had seen in the villages they passed. But for the past few days he had spoken to them about only one thing: the girl, the daughter of a merchant who lived in the village they would reach in about four days. He had been to the village
only once, the year before. The merchant was the proprietor of a dry goods shop, and he always demanded that the sheep be sheared in his presence, so that he would not be cheated. A friend had told the boy about the shop, and he had taken his sheep there. * "I need to sell some wool," the boy told the merchant. The shop was busy, and the man asked the shepherd to wait until the afternoon. So the boy sat on the steps of the shop and took a book from his bag.
"I didn't know shepherds knew how to read," said a girl's voice behind him. The girl was typical of the region of Andalusia, with flowing black hair, and eyes that vaguely recalled the Moorish conquerors. "Well, usually I learn more from my sheep than from books," he answered. During the two hours that they talked, she told him she was the merchant's daughter, and spoke of life in the village, where each day was like all the others. The shepherd told her of the Andalusian countryside, and related the news from the other towns where he had stopped.
It was a pleasant change from talking to his sheep. "How did you learn to read?" the girl asked at one point. "Like everybody learns," he said. "In school." "Well, if you know how to read, why are you just a shepherd?" The boy mumbled an answer that allowed him to avoid responding to her question. He was sure the girl would never understand. He went on telling stories about his travels, and her bright, Moorish eyes went wide with fear and surprise. As the time passed, the boy
found himself wishing that the day would never end, that her father would stay busy and keep him waiting for three days. He recognized that he was feeling something he had never experienced before: the desire to live in one place forever. With the girl with the raven hair, his days would never be the same again. But finally the merchant appeared, and asked the boy to shear four sheep. He paid for the wool and asked the shepherd to come back the following year."""

sentences = sent_tokenize(paragraph)
stemmer = PorterStemmer()

#stemming
for i in range(len(sentences)):
    words = word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
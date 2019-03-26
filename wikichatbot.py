'''
https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e
https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
https://github.com/philgyford/pepysdiary/blob/master/pepysdiary/encyclopedia/wikipedia_fetcher.py
Using the general concepts found here, we are going to make a nlp-based chatbot
which takes the chat queries from the users, and looks up answers on wikipedia

We don't want to download everything and store it as files, so we want to scrape
wikipedia.

https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

We will use the information here to make it more contextual

to start, make sure we can use cosine similarity (at first, we'll work on
stuff like tensorflow and neural nets next) to compare the user's input
to that of the "expected" responses so that it can be less restrictive.
More human friendly!

'''
import random
import nltk
import numpy as np
import random
import string
from bs4 import BeautifulSoup
#import urllib2
import re
import bleach
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from dateutil.parser import parse
#from urllib.request import urlopen

#page = urlopen(inputpage)

#soup = BeautifulSoup(page, "html.parser")
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# print(remove_punct_dict)


def cleanHtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext
    # return raw_html


def getHtml(topic):
    """
    Passed the name of a Wikipedia page (eg, 'Samuel_Pepys'), it fetches
    the HTML content (not the entire HTML page) and returns it.
    Returns a dict with two elements:
        'success' is either True or, if we couldn't fetch the page, False.
        'content' is the HTML if success==True, or else an error message.
    """
    error_message = ''
    topic = topic.replace(" ", "_")
    url = 'https://en.wikipedia.org/wiki/%s' % topic

    try:
        response = requests.get(url,
                                params={'action': 'render'}, timeout=30)
    except requests.exceptions.ConnectionError:
        error_message = "Can't connect to domain."
    except requests.exceptions.Timeout:
        error_message = "Connection timed out."
    except requests.exceptions.TooManyRedirects:
        error_message = "Too many redirects."

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        # 4xx or 5xx errors:
        error_message = "HTTP Error: %s" % response.status_code
    except NameError:
        if error_message == '':
            error_message = "Unknown Error Occurred"

    if error_message:
        return {'success': False, 'content': error_message}
    else:
        return {'success': True, 'content': response.text}


def tidyHtml(html):
    html = bleachHtml(html)
    html = stripHtml(html)
    html = cleanHtml(html)
    return html


def bleachHtml(html):

    # Pretty much most elements, but no forms or audio/video.
    allowed_tags = [
        'abbr', 'acronym', 'address', 'area', 'article',
        'b', 'blockquote', 'br',
        'code', 'col', 'colgroup',
        'dd', 'del', 'dfn', 'div', 'dl', 'dt',
        'em',
        'footer',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hgroup', 'hr',
        'i', 'img', 'ins',
        'kbd',
        'li',
        'map',
        'nav',
        'ol',
        'p', 'pre',
        'q',
        's', 'samp', 'section', 'small', 'span', 'strong', 'sub', 'sup',
        'table', 'tbody', 'td', 'tfoot', 'th', 'thead', 'time', 'tr',
        'ul',
        'var',
        # We allow script here, so we can close/un-mis-nest its tags,
        # but then it's removed completely in _strip_html():
        'script',
    ]

    # These attributes will be removed from any of the allowed tags.
    allowed_attributes = {
        '*':        ['class', 'id'],
        'a':        ['href', 'title'],
        'abbr':     ['title'],
        'acronym':  ['title'],
        'img':      ['alt', 'src', 'srcset'],
        # Ugh. Don't know why this page doesn't use .tright like others
        # http://127.0.0.1:8000/encyclopedia/5040/
        'table':    ['align'],
        'td':       ['colspan', 'rowspan'],
        'th':       ['colspan', 'rowspan', 'scope'],
    }

    return bleach.clean(html, tags=allowed_tags,
                        attributes=allowed_attributes, strip=True)


def stripHtml(html):
    """
    Takes out any tags, and their contents, that we don't want at all.
    And adds custom classes to existing tags (so we can apply CSS styles
    without having to multiply our CSS).
    Pass it an HTML string, it returns the stripped HTML string.
    """

    # CSS selectors. Strip these and their contents.
    selectors = [
        'div.hatnote',
        'div.navbar.mini',  # Will also match div.mini.navbar
        # Bottom of https://en.wikipedia.org/wiki/Charles_II_of_England :
        'div.topicon',
        'a.mw-headline-anchor',
        'script',
    ]

    # Strip any element that has one of these classes.
    classes = [
        # "This article may be expanded with text translated from..."
        # https://en.wikipedia.org/wiki/Afonso_VI_of_Portugal
        'ambox-notice',
        'magnify',
        'citation*',
        "reference"
        # eg audio on https://en.wikipedia.org/wiki/Bagpipes
        'mediaContainer',
        'navbox',
        'noprint',
    ]
    # strip completely these unwanted tags from the html
    tags_to_strip = [
        'h1',
        'h2',
        'h3',
        'h4',
        'h5',
        'h6',
        'caption',
        'cite',
        'figcaption',
        'figure',
        'footer',
        'a',
        'li',
        'img',
        'ol'
    ]

    # Any element has a class matching a key, it will have the classes
    # in the value added.
    add_classes = {
        # Give these tables standard Bootstrap styles.
        'infobox':   ['table', 'table-bordered'],
        'ambox':     ['table', 'table-bordered'],
        'wikitable': ['table', 'table-bordered'],
    }

    soup = BeautifulSoup(html, 'html5lib')

    for striptag in tags_to_strip:
        [tag.decompose() for tag in soup.find_all(striptag)]

    for selector in selectors:
        [tag.decompose() for tag in soup.select(selector)]

    for clss in classes:
        [tag.decompose() for tag in soup.find_all(attrs={'class': clss})]

    # for clss, new_classes in add_classes.items():
    #    for tag in soup.find_all(attrs={'class': clss}):
    #        tag['class'] = tag.get('class', []) + new_classes

    # Depending on the HTML parser BeautifulSoup used, soup may have
    # surrounding <html><body></body></html> or just <body></body> tags.
    if soup.body:
        soup = soup.body
    elif soup.html:
        soup = soup.html.body

    # Put the content back into a string.
    html = ''.join(str(tag) for tag in soup.contents)

    return html


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def tokenchecker(tokens, user_response):
    tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    allindex = list(reversed(vals.argsort()[0]))[1:6]
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    tokens.remove(user_response)
    returnarray = [allindex, idx, req_tfidf]
    return returnarray


def thanking():
    thankarray = ["You're welcome", "Absolutely!", "Happy to help!"]
    return random.choice(thankarray)


def greetings():
    greetingsarray = ["Hello!", "How can I help?", "hi", "hey", "yes?",
                      "hello", "What can I do for you?"]
    return random.choice(greetingsarray)


def goodbye():

    return '\*_'  # basically something which will never be printed


def picknew():

    return '\*\*_'  # basically something which will never be printed 


def goodbyeresponses():
    byearray = ["See you soon!", "Goodbye!", "Bye! Take Care!", ]
    return random.choice(byearray)


def parsingresponse(sent_tokens, req_tfidf, allindex, idx):
    robo_response = ""
    print("I have a confidence of ")
    print(req_tfidf)
    # we should narrow our response if there is a high confidence,
    # and expand our response if there is a low one.
    print("for this response")
    if(req_tfidf == 0):
        robo_response = robo_response + "Sorry, I can't find anything \
        in the article related to that..."
        return robo_response
    else:

        if(req_tfidf >= .75):
            print("I'm very confident on my response here")
            robo_response += sent_tokens[idx]
        elif(req_tfidf < .75 and req_tfidf >= .3):
            print("I'm not so sure, so I'll provide more context")
            try:
                robo_response += sent_tokens[idx - 1]
            except:
                pass
            robo_response += sent_tokens[idx]
            try:
                robo_response += sent_tokens[idx + 1]
            except:
                pass
        else:
            print("I'm really not confident with this response, so I'll")
            print("provide as much context as I can.")
            try:
                robo_response += sent_tokens[idx - 2]
            except:
                pass
            try:
                robo_response += sent_tokens[idx - 1]
            except:
                pass
            robo_response += sent_tokens[idx]
            try:
                robo_response += sent_tokens[idx + 1]
            except:
                pass
            try:
                robo_response += sent_tokens[idx + 2]
            except:
                pass
        return robo_response


def response(user_response, sent_tokens):
    new_topic_tokens = ['lets talk about something else', 'new topic',
                        'lets discuss a new topic',
                        'i want to discuss a new article',
                        'what else can we talk about']
    thanks_tokens = ['thanks', 'thank you']
    bye_tokens = ['bye', 'goodbye', 'see you later', ]
    greetings_tokens = ["hello", "hi",
                        "greetings", "sup", "what's up", "hey", ]
    robo_response = ''
    new_tfidf = tokenchecker(new_topic_tokens, user_response)[-1]
    print(new_tfidf)
    thanks_tfidf = tokenchecker(thanks_tokens, user_response)[-1]
    print(thanks_tfidf)
    bye_tfidf = tokenchecker(bye_tokens, user_response)[-1]
    print(bye_tfidf)
    greetings_tfidf = tokenchecker(greetings_tokens, user_response)[-1]
    print(greetings_tfidf)

    sent_array = tokenchecker(sent_tokens, user_response)

    req_tfidf = sent_array[-1]


    fullarray = sent_array[0]

    index = sent_array[1]

    tfidf_dict = {picknew: new_tfidf,
                  thanking: thanks_tfidf,
                  goodbye: bye_tfidf,
                  greetings: greetings_tfidf,
                  parsingresponse: req_tfidf}

    functiontouse = None
    # if there are duplicates, then the dictionary will be less than 5.
    # if that's the case, default to the parsingresponse value.
    # that one already has a "not sure" catchall.

    # that logic doesn't work. there may be an exact match, leaving a shortened
    # dictionary
    # print(len(tfidf_dict))
    if (max(tfidf_dict.values()) > 0):
        counter = 0
        for key in tfidf_dict:
            if (tfidf_dict[key] == max(tfidf_dict.values())):
                counter += 1
        if counter == 1:
            functiontouse = list(tfidf_dict.keys())[list(
                tfidf_dict.values()).index(max(tfidf_dict.values()))]
            if not (req_tfidf==max(tfidf_dict.values())):
                print("we are here")
                print(str(functiontouse))
                #functiontouse = tfidf_dict[max(tfidf_dict.values())]
                return functiontouse()
            else:
                return functiontouse(sent_tokens, req_tfidf, fullarray, index)
        else:
            print(sent_tokens)
            print(req_tfidf)
            return parsingresponse(sent_tokens, req_tfidf, fullarray, index)

    else:
        print(sent_tokens)
        print(req_tfidf)
        print(fullarray)
        print(index)
        return parsingresponse(sent_tokens, req_tfidf, fullarray, index)


'''
we will start with simple: in order to ask questions about a topic,
the user must supply the url to the page. The page will be parsed
and then we can start the chatbot

'''


def choosetopic():
    prompts = ["What topic would you like to cover? > ",
               "What would you like to talk about? > ",
               "What topic would you like to chat about? > ", ]
    inputpage = input(random.choice(prompts))
    result = getHtml(inputpage)
    pagehtml = None
    if result['success']:
        pagehtml = tidyHtml(result['content'])
    return [pagehtml, inputpage]


# we now have our page downloaded, and we have the html cleaned to
# purely text. Now we can parse it!

def returntokens(html):
    readuparray = ["Okay, let me read up on that real quick...",
                   "Processing, stand by...",
                   "*Jeopardy! theme* ...",
                   "Thanks! Let me freshen up on that topic...", ]
    print(random.choice(readuparray))
    raw = str(html).lower().replace('\n', ' ')
    nltk.download('punkt')  # first-time use only
    nltk.download('wordnet')  # first-time use only
    sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
    word_tokens = nltk.word_tokenize(raw)  # converts to list of words
    return [sent_tokens, word_tokens]

# we now have our word tokens.


def initializequestion():

    pagedata = choosetopic()
    pagehtml = pagedata[0]
    inputpage = pagedata[1]

    tokenset = returntokens(pagehtml)
    print("I've read up on the wikipedia article about " +
          inputpage + ". Feel free to ask questions about it!" +
          " If you want to exit, type Bye, and if you want to change topics," +
          " type 'let's discuss a new topic'.")

    return tokenset


def main():

    flag = True
    tokenset = initializequestion()
    sentence_tokens = tokenset[0]
    word_tokens = tokenset[1]
    while(flag == True):
        user_response = input("> ")
        user_response = user_response.lower()
        chatbotresponse = response(user_response, sentence_tokens)
        if (chatbotresponse == '\*_'):
            flag = False
            print(goodbyeresponses)
        if (chatbotresponse == '\*\*_'):
            tokenset = initializequestion()
            sentence_tokens = tokenset[0]
            word_tokens = tokenset[1]
        else:
            print(chatbotresponse)


if (__name__ == "__main__"):
    main()

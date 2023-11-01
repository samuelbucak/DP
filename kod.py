import requests
import json
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import re
import webbrowser
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from bokeh.io import show
from bokeh.plotting import from_networkx, figure
from bokeh.models import Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, ColumnDataSource, Text
from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

def map_nodes_to_integers_with_labels(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    H = nx.relabel_nodes(G, mapping)
    for node, label in mapping.items():
        H.nodes[label]['name'] = node
    return H

def showInteractiveMM(G):
    G = map_nodes_to_integers_with_labels(G)

    #Pridelenie váhy hránam na základe ich dôležitosti
    for u, v, d in G.edges(data=True):
        d['weight'] = G.degree(u) + G.degree(v)

    #Vytvorenie grafu
    plot = figure(width=800, height=800, x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
    plot.title.text = "Interactive Mind Map"

    #Vytvorenie bokeh grafu z networkx grafu
    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
    graph_renderer.node_renderer.data_source.data['name'] = [G.nodes[node]['name'] for node in G.nodes()]

    #Zablokovanie defaultneho renderera hran
    graph_renderer.edge_renderer.visible = False

    plot.renderers.append(graph_renderer)

    #Manuálne priradenie hrúbky hránam
    for start, end, data in G.edges(data=True):
        xs, ys = zip(*[(x, y) for x, y in [graph_renderer.layout_provider.graph_layout[start], graph_renderer.layout_provider.graph_layout[end]]])
        plot.line(xs, ys, line_width=data['weight'], color="#CCCCCC", alpha=0.8)

    #Pridanie nástrojov
    hover = HoverTool(tooltips=[("Name", "@name")])
    plot.add_tools(hover, TapTool(), BoxSelectTool())

    #Štýlovanie grafu
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    #Extrahovanie koordinatov z grafu
    node_coordinates = graph_renderer.layout_provider.graph_layout
    x_values = [x for x, _ in node_coordinates.values()]
    y_values = [y for _, y in node_coordinates.values()]
    names = [G.nodes[node]['name'] for node in G.nodes()]
    source = ColumnDataSource(data=dict(x=x_values, y=y_values, name=names))
    labels = Text(x='x', y='y', text='name', text_align='center', text_baseline='middle')
    plot.add_glyph(source, labels)

    #Zobrazenie grafu
    show(plot)
    
    #TO DO Pridanie callback funkcie na kliknutie na uzol

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    #Vytvorí zoznam kľúčových slov zo zoradených položiek
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    #Iterácia cez zoradené položky
    for idx, score in sorted_items:
        fname = feature_names[idx]
        #Získanie názvu funkcie a skóre
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    #Iterácia cez zoznam kľúčových slov a skóre
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return list(results.keys())

def processTextToKeywords(text):
    #Tokenizovanie textu do viet
    sentences = nltk.sent_tokenize(text)
    #Tokenizovanie viet do slov
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    #Vytvorenie bigramov
    bigram = Phrases(tokenized_sentences, min_count=1, threshold=1)
    #Vytvorenie trigramov
    trigram = Phrases(bigram[tokenized_sentences], min_count=1, threshold=1)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    #Aplikácia bigramov a trigramov na tokenizované vety
    processed_text = [' '.join(trigram_mod[bigram_mod[sentence]]) for sentence in tokenized_sentences]

    #Spojenie spracovaných viet do jedného textového reťazca
    joined_text = ' '.join(processed_text)

    #Vytvorenie TF-IDF vektorizéra
    vectorizer = TfidfVectorizer()
    #Výpočet TF-IDF hodnôt
    tfidf_matrix = vectorizer.fit_transform([joined_text])
    #Získanie názvov funkcií a zoradenie podľa dôležitosti
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sort_coo(tfidf_matrix.tocoo())

    #Získanie top 10 kľúčových slov
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    return keywords

def determineInputType(input_text):
    #Určenie typu vstupu
    if "commit" in input_text and "Author" in input_text:
        return "git"
    elif len(input_text.split()) > 10: #Ak je vstup dlhší ako 10 slov, považuje sa za dokumentáciu
        return "doc"
    else:
        #Kratšie vstupy sa považujú za kľúčové slová
        return "keywords"

# Predpokladá sa, že každý riadok je vo formáte '<hash> <commit message>' (Používateľ to dosiahne príkazom 'git log --pretty=oneline')
def processGitLog(git_log_text):
    #Rozdelenie git log textu na riadky
    lines = git_log_text.strip().split('\n')
    #Inicializácia zoznamu commit správ
    commit_messages = []
    #Iterácia cez každý riadok a extrakcia commit správ
    for line in lines:
        #Môžeme použiť regulárny výraz na extrakciu commit správ
        match = re.match(r'^[a-fA-F0-9]+\s+(.*)$', line)
        if match:
            commit_message = match.group(1)
            commit_messages.append(commit_message)
    return commit_messages

def userInput():
    try:
        with open("keywords.txt", 'r', encoding='utf-8') as file:
            input_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    input_type = determineInputType(input_text)
    keywords = []
    if input_type == "git":
        #Spracovanie git logu na získanie commit správ
        commit_messages = processGitLog(input_text)
        #Zlúčenie commit správ do jedného textového reťazca
        text = '. '.join(commit_messages)
        #Spracovanie textu na získanie kľúčových slov
        keywords = processTextToKeywords(text)
    elif input_type == "doc":
        keywords = processTextToKeywords(input_text)
    else:
        keywords = input_text.strip().split()

    print(f"Kľúčové slová: {keywords}")
    return keywords

def searchSO(keywords, numberOfQuestions=20):
    questions = []
    while len(questions) < numberOfQuestions and len(keywords) > 0:
        query = '+'.join(keywords).replace(' ', '+').replace('_', ' ')
        url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={query}&site=stackoverflow&filter=withbody"
        response = requests.get(url)
        if response.status_code == 200:
            results = json.loads(response.text)
            for question in results['items'][:numberOfQuestions - len(questions)]:
                #Získanie odpovede pre aktuálnu otázku
                answer_url = f"https://api.stackexchange.com/2.3/questions/{question['question_id']}/answers?site=stackoverflow&filter=withbody"
                answer_response = requests.get(answer_url)
                if answer_response.status_code == 200:
                    answers = json.loads(answer_response.text)['items']
                    #Triedenie odpovedí podľa počtu upvotov
                    top_answers = sorted(answers, key=lambda x: x['score'], reverse=True)[:5]                    
                    question['answers'] = [answer['body'] for answer in top_answers if 'body' in answer]
                questions.append(question)
        if len(questions) < numberOfQuestions:
            keywords.pop() #odstránenie posledného kľúčového slova
    return questions
    
def nounsVerbs(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return [word for word, pos in tagged if pos.startswith('NN') or pos.startswith('VB')]

def createMM(questions):
    #Načítanie stopwords
    stop_words = set(stopwords.words('english'))

    G = nx.Graph()

    for question in questions:
        title = question['title']
        words = nounsVerbs(title)
        for word in words:
            word = word.lower()
            if word not in stop_words:
                G.add_node(word)
                for other_word in words:
                    other_word = other_word.lower()
                    if other_word != word and other_word not in stop_words:
                        if not G.has_edge(word, other_word):
                            #Ak medzi slovami neexistuje hrana, vytvorí ju s váhou 1
                            G.add_edge(word, other_word, weight=1)
                        else:
                            #Ak medzi slovami existuje hrana, zvýši jej váhu o 1
                            G[word][other_word]['weight'] += 1

    #Ak sú medzi tromi uzlami viac ako 2 hrany, odstráni hrany s najnižšou váhou
    for u, v in list(G.edges()):
        for w in G.nodes():
            if w!=u and w!=v:
                weights = []
                if G.has_edge(u, w):
                    weights.append(G[u][w]['weight'])
                if G.has_edge(v, w):
                    weights.append(G[v][w]['weight'])
                if G.has_edge(u, v):
                    weights.append(G[u][v]['weight'])

                if len(weights) == 3:
                    min_weight = min(weights)
                    if G.has_edge(u, w) and G[u][w]['weight'] == min_weight:
                        G.remove_edge(u, w)
                    elif G.has_edge(v, w) and G[v][w]['weight'] == min_weight:
                        G.remove_edge(v, w)
                    elif G.has_edge(u, v) and G[u][v]['weight'] == min_weight:
                        G.remove_edge(u, v)
    
    return G

def generateHTML(questions):
    html_content = "<html><head><title>StackOverflow Q&A</title></head><body>"

    for idx, question in enumerate(questions, 1):
        html_content += f"<h2>Question {idx}: {question['title']}</h2>"
        html_content += f"<div>{question['body']}</div>"
        if 'answers' in question:
            for answer_idx, answer in enumerate(question['answers'], 1):
                html_content += f"<h3>Answer {answer_idx}:</h3><div>{answer}</div>"
        html_content += "<hr>"
    html_content += "</body></html>"

    return html_content

def printQuestions(questions):
    #Vygenerovanie HTML obsahu
    html_content = generateHTML(questions)

    #Uloženie do súboru
    temp_file = "temp_questions_answers.html"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    #Otvorenie súboru v prehliadači
    webbrowser.open('file://' + os.path.realpath(temp_file))

def main():
    #Získanie vstupu od používateľa
    keywords = userInput()

    #Ak je zoznam prázdny
    if not keywords:
        print("No results")
        return
    
    #Vyhľadanie otázok na StackOverflow
    questions = searchSO(keywords)

    if questions:
        printQuestions(questions)
        G = createMM(questions)
        showInteractiveMM(G)
    else:
        print("No results")

if __name__ == "__main__":
    main()
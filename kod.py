import requests
import json
import networkx as nx
import nltk
from bokeh.models.widgets import Button
from bokeh.layouts import column
from nltk.corpus import stopwords
from bokeh.plotting import from_networkx, figure
from bokeh.models import Range1d, Circle, HoverTool, TapTool, BoxSelectTool, ColumnDataSource, Text, TextInput, Button, Div
from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

def map_nodes_to_integers_with_labels(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    H = nx.relabel_nodes(G, mapping)
    for node, label in mapping.items():
        H.nodes[label]['name'] = node
    return H

def modify_doc(doc):
    #Používateľský vstup
    text_input = TextInput(value="Enter keywords here")
    search_button = Button(label="Search", button_type="success")
    result_div = Div(text="")
    # Inicializácia prázdneho miesta pre graf
    placeholder = Div(text="Graf sa načíta po vykonaní vyhľadávania")
    layout = column(text_input, search_button, result_div, placeholder)

    def search_callback():
        keywords = text_input.value.split(",") #Kľúčové slová oddelené čiarkou

        #Vyhľadávanie
        questions = searchSO(keywords)

        if questions:
            html_content = generateHTML(questions)
            result_div.text = html_content
            G = createMM(questions)
            #VYTVORENIE GRAFU
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

            #Normalizovanie hrúbok hrán
            max_weight = max([data['weight'] for _, _, data in G.edges(data=True)])
            scaled_weights = [(data['weight'] / max_weight) * 2.5 + 2.5 for _, _, data in G.edges(data=True)]

            #Manuálne priradenie hrúbky hránam
            for (start, end, data), width in zip(G.edges(data=True), scaled_weights):
                xs, ys = zip(*[(x, y) for x, y in [graph_renderer.layout_provider.graph_layout[start], graph_renderer.layout_provider.graph_layout[end]]])
                plot.line(xs, ys, line_width=width, color="#CCCCCC", alpha=0.8)

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
            #Poziciovanie textu nad uzly
            y_values = [y + 0.05 for _, y in node_coordinates.values()]
            names = [G.nodes[node]['name'] for node in G.nodes()]
            source = ColumnDataSource(data=dict(x=x_values, y=y_values, name=names))
            labels = Text(x='x', y='y', text='name', text_align='center', text_baseline='middle')
            plot.add_glyph(source, labels)
            # Aktualizácia layoutu s novým grafom
            layout.children[-1] = plot  # Nahrádzame posledný element (placeholder) grafom
        else:
            result_div.text = "No Results"
    
    search_button.on_click(search_callback)

    # Pridanie layoutu do dokumentu
    doc.add_root(layout)


def update_document_with_graph(doc, G):
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

    #Normalizovanie hrúbok hrán
    max_weight = max([data['weight'] for _, _, data in G.edges(data=True)])
    scaled_weights = [(data['weight'] / max_weight) * 2.5 + 2.5 for _, _, data in G.edges(data=True)]

    #Manuálne priradenie hrúbky hránam
    for (start, end, data), width in zip(G.edges(data=True), scaled_weights):
        xs, ys = zip(*[(x, y) for x, y in [graph_renderer.layout_provider.graph_layout[start], graph_renderer.layout_provider.graph_layout[end]]])
        plot.line(xs, ys, line_width=width, color="#CCCCCC", alpha=0.8)

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
    #Poziciovanie textu nad uzly
    y_values = [y + 0.05 for _, y in node_coordinates.values()]
    names = [G.nodes[node]['name'] for node in G.nodes()]
    source = ColumnDataSource(data=dict(x=x_values, y=y_values, name=names))
    labels = Text(x='x', y='y', text='name', text_align='center', text_baseline='middle')
    plot.add_glyph(source, labels)

    #show(plot)

    """ def callback(event=None):
        selected_node_indices = graph_renderer.node_renderer.data_source.selected.indices
        selected_node_names = [graph_renderer.node_renderer.data_source.data['name'][i] for i in selected_node_indices]
        print("Selected nodes: ", selected_node_names)
        #spustenie programu znova TO DO
    
    button = Button(label="Run with selected nodes", button_type="success")
    button.on_click(callback)

    new_layout = layout([[button], [plot]]) """

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

# Vytvorenie Bokeh aplikácie
bokeh_app = Application(FunctionHandler(modify_doc))

def run_server():
    # Spustenie Bokeh servera
    server = Server({'/': bokeh_app}, port=5000)
    server.start()

    #Orvorenie aplikácie v prehliadači
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

if __name__ == "__main__":
    run_server()
import graphviz

def create_model_structure():
    dot = graphviz.Digraph(comment='MutationScorer Model Structure')
    dot.attr(rankdir='TB')
    
    # 设置图形样式
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray50', fontcolor='gray50')
    
    # 添加主节点
    dot.node('input', 'Input\n(Map, Commanders, Mutations, AI)', shape='box', fillcolor='lightgreen')
    
    # Feature Embedding 部分
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='FeatureEmbedding', style='rounded', color='blue', bgcolor='lightgray')
        c.node('map_emb', 'MapEmbedding\n[2, 48]')
        c.node('commander_emb', 'CommanderEmbedding\n[2, 48]')
        c.node('mutation_emb', 'MutationEmbedding\n[2, 48]')
        c.node('ai_emb', 'AIEmbedding\n[2, 48]')
        c.node('concat', 'Concatenate\n[2, 192]', fillcolor='lightyellow')
        
        # 连接嵌入层
        c.edge('map_emb', 'concat')
        c.edge('commander_emb', 'concat')
        c.edge('mutation_emb', 'concat')
        c.edge('ai_emb', 'concat')
    
    # MLP 部分
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='MLP', style='rounded', color='red', bgcolor='mistyrose')
        c.node('linear1', 'Linear\n[2, 128]')
        c.node('bn1', 'BatchNorm + ReLU + Dropout')
        c.node('linear2', 'Linear\n[2, 64]')
        c.node('bn2', 'BatchNorm + ReLU + Dropout')
        c.node('linear3', 'Linear\n[2, 32]')
        c.node('bn3', 'BatchNorm + ReLU + Dropout')
        c.node('output', 'Linear\n[2, 5]', fillcolor='lightpink')
        
        # 连接MLP层
        c.edge('linear1', 'bn1')
        c.edge('bn1', 'linear2')
        c.edge('linear2', 'bn2')
        c.edge('bn2', 'linear3')
        c.edge('linear3', 'bn3')
        c.edge('bn3', 'output')
    
    # 连接主要组件
    dot.edge('input', 'map_emb')
    dot.edge('input', 'commander_emb')
    dot.edge('input', 'mutation_emb')
    dot.edge('input', 'ai_emb')
    dot.edge('concat', 'linear1')
    
    # 保存图
    dot.render('model_structure', directory='experiments/model_viz', format='svg', cleanup=True)

if __name__ == '__main__':
    create_model_structure() 
import spacy

class SyntacticNode:
    def __init__(self, spacy_node=None, **kwargs):
        if spacy_node is not None:
            self.rel = spacy_node.dep_
            self.text = spacy_node.text
            self.pos = spacy_node.pos_
        else:
            self.rel = kwargs['rel']
            self.text = kwargs['text']
            self.pos = None
        self.label = None
        self.left = []
        self.right = []
        self.head = None
        self.head_direction = None

    def __str__(self):
        return self.text

    def add_left(self, left_node):
        self.left.append(left_node)
        left_node.head = self
        left_node.head_direction = 'left'

    def add_right(self, right_node):
        self.right.append(right_node)
        right_node.head = self
        right_node.head_direction = 'right'

    def get_subtext_list(self):
        result = []
        for item in self.left:
            result += item.get_subtext_list()
        result.append(self)
        for item in self.right:
            result += item.get_subtext_list()
        return result

    def get_check_list(self):
        result = []
        result.append(self)
        for item in self.left:
            result += item.get_check_list()
        for item in self.right:
            result += item.get_check_list()
        return result

    def list2text(self, subtext_list):
        return ' '.join([item.text for item in subtext_list])


class SyntacticTree:
    def __init__(self, text, nlp=None):
        if nlp == None:
            nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp
        doc = nlp(text)
        for item in doc:
            if item.dep_ != 'ROOT':
                continue
            else:
                root_node = item
                break
        self.head = SyntacticNode(root_node)
        process_list = [(self.head, root_node)]
        while(len(process_list) != 0):
            syntactic_node, spacy_node = process_list.pop()
            for item in spacy_node.lefts:
                syntactic_item = SyntacticNode(item)
                syntactic_node.add_left(syntactic_item)
                process_list.append((syntactic_item, item))
            for item in spacy_node.rights:
                syntactic_item = SyntacticNode(item)
                syntactic_node.add_right(syntactic_item)
                process_list.append((syntactic_item, item))
        #对实体词进行单独处理
        node_list = self.head.get_subtext_list()
        for entity in doc.ents:
            if [item for item in node_list[entity.start: entity.end] if
                item.head not in node_list[entity.start: entity.end]][0].head is None:
                break
            if None not in [item.head for item in node_list[entity.start: entity.end]]:
                head, direction, index = \
                    [(item.head, item.head_direction, item.head.__dict__[item.head_direction].index(item))
                     for item in node_list[entity.start: entity.end] if
                     item.head not in node_list[entity.start: entity.end]][0]
            else:
                head = direction = index = None
            for item in node_list[entity.start: entity.end]:
                self.delete_subtree(item)
            entity_node = SyntacticNode(text=entity.text, rel='entity')
            left_list = []
            for item in node_list[entity.start: entity.end]:
                for left_item in item.left:
                    if left_item not in node_list[entity.start: entity.end]:
                        left_item.head = entity_node
                        left_list.append(left_item)
            right_list = []
            for item in node_list[entity.start: entity.end]:
                for right_item in item.right:
                    if right_item not in node_list[entity.start: entity.end]:
                        right_item.head = entity_node
                        right_list.append(right_item)

            entity_node.label = entity.label_
            if head is not None:
                entity_node.head = head
                entity_node.head_direction = direction
                head.__dict__[direction].insert(index, entity_node)
            entity_node.left = left_list
            entity_node.right = right_list





    def get_sentence(self, target_p=None):
        if target_p is not None and target_p != self.head:
            node = target_p
            target_p_left = target_p.left
            target_p.left = []
            target_p_right = target_p.right
            target_p.right = []
            save_left_list = []
            save_right_list = []
            save_node_list = []
            while node.head is not None:
                head = node.head
                save_left = head.left
                save_right = head.right
                save_node_list.append(head)
                save_left_list.append(save_left)
                save_right_list.append(save_right)
                if node.head_direction == 'left':
                    node.head.left = []
                    node.head.right = []
                    for item in save_left:
                        node.head.left.append(item)
                        if item == target_p:
                            break
                if node.head_direction == 'right':
                    node.head.right = []
                    for item in save_right:
                        node.head.right.append(item)
                        if item == target_p:
                            break
                node = head
            text_list = self.head.get_subtext_list()
            text = self.head.list2text(text_list)
            for index in range(len(save_node_list)):
                save_node_list[index].left = save_left_list[index]
                save_node_list[index].right = save_right_list[index]
            target_p.left = target_p_left
            target_p.right = target_p_right
        else:
            text_list = self.head.get_subtext_list()
            text = self.head.list2text(text_list)
        return text

    def add_subtree(self, new_node, direction, node, index=None):
        if index is not None:
            node.__dict__[direction].insert(index, new_node)
        else:
            node.__dict__[direction].append(new_node)
        new_node.head = node
        new_node.head_direction = direction


    def replace_subtree_node(self, node, text):
        node.text = text

    def delete_subtree(self, node):
        if node.head_direction == 'left':
            for i in range(len(node.head.left)):
                if node.head.left[i] == node:
                    del node.head.left[i]
                    break
        if node.head_direction == 'right':
            for i in range(len(node.head.right)):
                if node.head.right[i] == node:
                    del node.head.right[i]
                    break

    def get_fore_node(self, node):
        node_list = self.head.get_subtext_list()
        if node not in node_list:
            return None
        else:
            index = node_list.index(node)
            if index == 0:
                return None
            else:
                return node_list[index-1]





if __name__ == "__main__":
    tree = SyntacticTree('Modern technology tries to restore the scene of World War II')
    a = tree.get_sentence()
    print(a)

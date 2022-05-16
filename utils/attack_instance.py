import json
import copy
from unittest import result
import csv

class AttackInstance:
    """
    Aa attacking example for detection.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, ground_truth, orig_text,orig_label,perd_text,perd_label,\
        orig_score, perd_score, result_type,num_queries,b_flag=False):
        self.ground = int(float(ground_truth))
        self.orig_text = orig_text
        self.orig_label= int(float(orig_label))
        self.perd_text = perd_text
        self.perd_label = int(float(perd_label))
        self.orig_score = float(orig_score)
        self.perd_score = float(perd_score)
        self.result_type = result_type
        self.num_queries = num_queries
        
        self.suss = False
        self.skip = False
        self.fail = False
        self.flag_update()

        self.atk_indices = []
        self.atk_changes = []
        #snli
        self.orig_text_b = ""
        self.perd_text_b = ""
        self.perd_trace(b_flag)
        
    def __repr__(self):
        return str(self.to_json_string())

    def flag_update(self):
        if self.result_type =='Failed':
            self.fail = True
        elif self.result_type == "Successful":
            self.suss = True
        else:
            self.skip = True

    def set_label(self, label: str):
        self.label = label

    def set_guid(self, guid: str):
        self.guid = guid

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def perturbable_sentence(self):
        if self.text_b is None:
            return self.text_a
        else:
            return self.text_b

    def is_nli(self):
        if self.text_b is None:
            return False
        else:
            return True

    def length(self):
        if self.text_b is None:
            return len(self.text_a.split())
        else:
            return len(self.text_b.split())
    def display(self):
        print("ground_truth_label: {}".format(self.ground))
        print("original_text: {}".format(self.orig_text))
        print("original_output: {}".format(self.orig_label))
        print("perturbed_text: {}".format(self.perd_text))
        print("perturbed_output: {}".format(self.perd_label))

    def deal_snli(self,tokens):

        temp = []
        for t in tokens:
            if "<SPLIT>" in t:
                t = t.split("<SPLIT>")
                temp.extend(t)
            else:
                temp.append(t)
        
        temp = [i[4:-5]+i[-1] if "[[[[" in i else i for i in temp]   
        return temp

    def perd_trace(self,b_flag):
        '''
        处理texts中的扰动标记, 并记录扰动的tokens
        '''
        input_tokens = self.orig_text.split(" ")
        perd_tokens = self.perd_text.split(" ")
        if b_flag==True:
            input_tokens = self.deal_snli(input_tokens)
            perd_tokens = self.deal_snli(perd_tokens)

        # print('input_tokens:{} \nperd_tokens:{}'.format(input_tokens,perd_tokens))
        # print("length: {},{}".format(len(input_tokens),len(perd_tokens)))
        assert(len(input_tokens)==len(perd_tokens))

        for i in range(0,len(input_tokens)):
            if "[[" in input_tokens[i] and input_tokens[i].index('[')==0 :
                assert("[[" in perd_tokens[i])
                # if "[[" not in perd_tokens[i]:
                #     print(f"errror for : {self.orig_text}")
                self.atk_indices.append(i)
                self.atk_changes.append([input_tokens[i].split('[[')[1].split(']]')[0],\
                                         perd_tokens[i].split('[[')[1].split(']]')[0]])
                input_tokens[i]=input_tokens[i].split('[[')[1].split(']]')[0]
                perd_tokens[i]=perd_tokens[i].split('[[')[1].split(']]')[0]
        self.orig_text = " ".join(input_tokens)
        self.perd_text = " ".join(perd_tokens)


def read_adv_files(file_path, b_flag = False):
    instances = []
    b_flag = False
    with open(file_path, mode='r') as csvf:
        csv_reader = csv.DictReader(csvf)
        line_count  = 0
        for line in csv_reader:
            line_count += 1            
            instances.append(AttackInstance(ground_truth=line['ground_truth_output'],\
                            orig_text=line['original_text'],orig_label=line['original_output'],\
                            perd_text=line['perturbed_text'], perd_label=line['perturbed_output'],\
                            orig_score=line['original_score'], perd_score=line['perturbed_score'], \
                            result_type=line['result_type'], num_queries=line['num_queries']
                            ,b_flag=b_flag))
    #logger.info("Load [{}] attack instance.".format(len(instances)))
    return instances

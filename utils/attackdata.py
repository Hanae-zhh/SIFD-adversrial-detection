def read_adv_files(file_path):
    instances = []
    b_flag = False
    if args.dataset == 'snli':
        b_flag = True
    with open(file_path, mode='r') as csvf:
        csv_reader = csv.DictReader(csvf)
        line_count  = 0
        for line in csv_reader:
            if line_count == 0:
                logger.info(f'------ names in csv file ------\n{", ".join(line)}')
            line_count += 1            
            instances.append(AttackInstance(ground_truth=line['ground_truth_output'],\
                            orig_text=line['original_text'],orig_label=line['original_output'],\
                            perd_text=line['perturbed_text'], perd_label=line['perturbed_output'],\
                            orig_score=line['original_score'], perd_score=line['perturbed_score'], \
                            result_type=line['result_type'], num_queries=line['num_queries']
                            ,b_flag=b_flag))
    logger.info("Load [{}] attack instance.".format(len(instances)))
    return instances
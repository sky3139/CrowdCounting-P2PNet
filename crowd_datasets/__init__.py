# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data
    if args.dataset_file == 'PRCV':
        from crowd_datasets.SHHA.loading_data import loading_dataPRCV
        return loading_dataPRCV
    return None
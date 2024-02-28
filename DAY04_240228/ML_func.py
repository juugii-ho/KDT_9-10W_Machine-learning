def fDFcolumns (old, feature, data, new):
    new = old[old[feature] == data]
    new = new.reset_index(drop=True)
    return new



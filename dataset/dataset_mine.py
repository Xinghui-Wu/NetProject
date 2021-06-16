import snap

amazon_path='com-amazon.ungraph.txt'
dblp_path='com-dblp.ungraph.txt'

cit_hepPh_path='cit-HepPh.txt'
cit_hepPh_dates_path='cit-HepPh-dates.txt'

cit_hepTh_path='cit-HepTh.txt'
cit_hepTh_dates_path='cit-HepTh-dates.txt'
cit_hepTh_abstracts_path='cit-HepTh-abstracts.tar.gz'



# G_amazon= snap.LoadEdgeList(snap.PNGraph,amazon_path,0,1)
# G_dblp=snap.LoadEdgeList(snap.PNGraph,dblp_path,0,1)
# G_citHepTh=snap.LoadEdgeList(snap.PNGraph,cit_hepTh_path,0,1)
# G_citHepPh=snap.LoadEdgeList(snap.PNGraph,cit_hepPh_path,0,1)



def G_return(path):

    return snap.LoadEdgeList(snap.PNGraph,path,0,1)

if __name__ == '__main__':
    G_return(amazon_path)
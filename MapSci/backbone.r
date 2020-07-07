library(disparityfilter)
library(RcppCNPy)

args = commandArgs(trailingOnly=TRUE)
if (length(args)<=2)
{
	stop("Missing parameters.", call.=FALSE)
}

phi <- npyLoad(args[1])
alpha <- as.numeric(args[2])
g <- graph.adjacency(phi, mode="undirected", weighted=TRUE)
E <- as.matrix(backbone(g, alpha=alpha)[,c("from", "to")])
h <- graph_from_edgelist(E, directed=F)
print(args[3])
write_graph(h, args[3], format="edgelist")

# PLOT
#pdf("backbone3.pdf", 7, 7)
#plot(hss, vertex.size=5, edge.eidth=0.5, edge.arrow.size=0,
#	vertex.color=vals,vertex.label=NA,
#   main=paste("StarSpace alpha=0.025\nEdges =",
#	length(E.ss[E.ss[,"alpha"] < 0.01, "alpha"])))
#legend("topleft", legend=c("health", "life", "physical", "social"),
#	pch=21, pt.bg=c(categorical_pal(4)))
#dev.off()

library(disparityfilter)
library(RcppCNPy)

# Sorry for these variable names. It is quite simple tho.

# Input reading
args = commandArgs(trailingOnly=TRUE)
if (length(args)<=2)
{
	stop("Missing parameters.", call.=FALSE)
}
phi <- npyLoad(args[1])
alpha <- as.numeric(args[2])

# Get your backbone
g <- graph.adjacency(phi, mode="undirected", weighted=TRUE)
E <- backbone(g, alpha=alpha, directed=F)[,c("from", "to")]
mn <- pmin(E$from, E$to)
mx <- pmax(E$from, E$to)
int <- as.numeric(interaction(mn, mx))
E <- as.matrix(E[match(unique(int), int),])

# Edges inside groups vs between groups
h <- graph_from_edgelist(E, directed=F)
c <- membership(cluster_fast_greedy(h))
for(i in 1:(dim(E)[1]))
{
	if(c[E[i,1]] == c[E[i,2]])
	{
		E(h)[i]$weight <- 1 
	}
	else
	{
		E(h)[i]$weight <- 2
	}
}

# And it's done
print(args[3])
write_graph(h, args[3], format="ncol", weight="weight")

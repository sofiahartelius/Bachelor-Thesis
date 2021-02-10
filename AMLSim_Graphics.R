# Only frauds

#### Restore Session & set major parameters ####
this.dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(this.dir)

# Restore Session & set major parameters #
cat("\014") 
rm(list=ls())
graphics.off()

# Restore 
this.dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(this.dir)

print("Session Restored & new parameters set")
#### Packages ####
library(tidyverse)
library(ggplot2)
library(MASS)
library(chron)
library(scatterplot3d)
library(cowplot)
library(GGally)
library(vioplot)
library(leaps)
library(ISLR)
library(igraph)
library(wesanderson)
library(extrafont)
loadfonts()

#### Load data
accounts <- read.csv("./100vertices-10Kedges/accounts.csv", 
                  header = TRUE, 
                  stringsAsFactors = FALSE)
transactions <- read.csv("./100vertices-10Kedges/transactions.csv", 
                         header = TRUE, 
                         stringsAsFactors = FALSE)
alerts <- read.csv("./100vertices-10Kedges/alerts.csv", 
                         header = TRUE, 
                         stringsAsFactors = FALSE)


# Fetch falsely accused nodes
p <- c(1,15,21,22,35,36,37,73,75,80,83,84,85,86,92,96,98)

poor_nodes <- accounts[p,1]


# Remove non-frauds
accounts <- accounts[-which(accounts$IS_FRAUD == "false"),]
accounts$test <- c(1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,0,1)

print(c(accounts$ACCOUNT_ID))

# Build AMLSim100 full graph
links_0 <- transactions[,- c(1, 4:8)]



G_0 <- graph_from_data_frame(links_0, directed = F)
G_0 <- simplify(G_0, remove.multiple = F)

par(mfrow = c(1,1))
plot(G_0,
     layout=layout.circle,
     main = "AMLSim100 Graph",
     vertex.color = wesanderson::wes_palettes$Royal1[1],
     vertex.size=5,
     vertex.frame.color= "white", #wesanderson::wes_palettes$BottleRocket2[4],
     vertex.label.family = "Bodoni MT",
     vertex.label.color="white",
     vertex.label.font = 2,
     vertex.label.cex=0.8,
     edge.curved=0,
     edge.color=wesanderson::wes_palettes$BottleRocket2[4],
     edge.width = 0.3)



#$IS_FRAUD <- accounts$IS_FRAUD

#G <- graph_from_data_frame(transactions[,- c(1, 4:8)], directed = F)
#V(G)$IS_FRAUD <- accounts$IS_FRAUD
#G <- simplify(G, remove.multiple = F)
#plot(G,layout=layout.fruchterman.reingold, vertex.color=colrs[1+(V(G)$IS_FRAUD == "T"]))
#G <- simplify(G, remove.multiple = T)
#plot(G,layout=layout.fruchterman.reingold, vertex.color=c( "gold", "red")[1+(V(G)$IS_FRAUD=="true")])


sub <- transactions[transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[1] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[2] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[3] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[4] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[5] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[6] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[7] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[8] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[9] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[10] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[11] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[12] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[13] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[14] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[15] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[16] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[17] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[18] |
                      transactions$SENDER_ACCOUNT_ID == accounts$ACCOUNT_ID[19] ,]

table(sub$SENDER_ACCOUNT_ID)

sub <- sub[sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[1] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[2] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[3] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[4] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[5] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[6] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[7] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[8] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[9] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[10] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[11] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[12] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[13] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[14] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[15] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[16] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[17] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[18] |
                      sub$RECEIVER_ACCOUNT_ID == accounts$ACCOUNT_ID[19] ,]

sub <- sub[,- c(1, 4:6, 8)]
links_1 <- sub[,-3]
links_2 <- sub[which(sub$IS_FRAUD == "True"),]
links_2 <- links_2[,-3]

G_1 <- graph_from_data_frame(links_1, directed = F)
G_1 <- simplify(G_1, remove.multiple = F)

G_2 <- graph_from_data_frame(links_2, directed = T)
G_2 <- simplify(G_2, remove.multiple = T)

colrs <- c("red", "blue")
V(G_2)$color <- colrs[V(G_2)$test]

par(mfrow = c(1,1))
plot(G_1,
     layout=layout.circle,
     main = "AMLSim100 Graph",
     vertex.color = wesanderson::wes_palettes$Royal1[1],
     vertex.size=20,
     vertex.frame.color= "white", #wesanderson::wes_palettes$BottleRocket2[4],
     vertex.label.family = "Bodoni MT",
     vertex.label.color="white",
     vertex.label.font = 2,
     vertex.label.cex=1.8,
     edge.curved=0,
     edge.color=wesanderson::wes_palettes$BottleRocket2[4],
     edge.width = 3)

par(mar = c(2,3,5,3), cex.main = 1.8 )
set.seed(2020)
plot(G_2,
     layout=layout.fruchterman.reingold,
     main = "Fraudulent Networks",
     main.family = "Comic Sans MS",
     vertex.color = wesanderson::wes_palettes$Royal1[1],
     vertex.size=20,
     vertex.frame.color= "white", #wesanderson::wes_palettes$BottleRocket2[4],
     vertex.label.family = "Bodoni MT",
     vertex.label.font = 2,
     vertex.label.color="white",
     vertex.label.cex=1.8,
     edge.curved=0,
     edge.color=wesanderson::wes_palettes$BottleRocket2[4],
     edge.width = 3)

# [1] 55 53 66 67 90 24 9  42 95 17 74 57 81 43 82 39 20 12 47

V(G_2)$test <- c(1,0,1,1,1,1,0,0,0,1,1,1,2,2,1,1,1,2,2)
set.seed(2020)
plot(G_2,
     layout=layout.fruchterman.reingold,
     main = "GCN Detection Rate",
     vertex.color = ifelse(V(G_2)$test == 1,
                           wesanderson::wes_palettes$Royal1[2],
                           ifelse(V(G_2)$test == 0,
                                  wesanderson::wes_palettes$Royal1[1],
                                  wesanderson::wes_palettes$Moonrise3[3])),
     vertex.size=20,
     vertex.frame.color= "white", #wesanderson::wes_palettes$BottleRocket2[4],
     vertex.label.family = "Bodoni MT",
     vertex.label.color="white",
     vertex.label.font = 2,
     vertex.label.cex=1.8,
     edge.curved=0,
     edge.color=wesanderson::wes_palettes$BottleRocket2[4],
     edge.width = 3)
legend( "bottomright",
        legend = c("Detected", "Undetected", "Not present"), 
        fill = c(wesanderson::wes_palettes$Royal1[2],
                                                   wesanderson::wes_palettes$Royal1[1],
                                                   wesanderson::wes_palettes$Moonrise3[3]),
        bg = "white",
        inset = 0,
        cex = 1.9,
        title = "Legend",
        box.lty=0)

G_3 <- make_empty_graph() + 
        vertices(poor_nodes)
V(G_3)$test <- rep(1,17)

set.seed(2020)
plot(G_3,
     layout=layout.fruchterman.reingold,
     main = "Falsely Accused Nodes",
     vertex.color = wesanderson::wes_palettes$Royal1[2],
     vertex.size=20,
     vertex.frame.color= "white", #wesanderson::wes_palettes$BottleRocket2[4],
     vertex.label.family = "Bodoni MT",
     vertex.label.color="white",
     vertex.label.font = 2,
     vertex.label.cex=1.8,
     edge.curved=0,
     edge.color=wesanderson::wes_palettes$BottleRocket2[4],
     edge.width = 3)

#layouts <- grep("^layout_", ls("package:igraph"), value=TRUE)[-1] 

#Remove layouts that do not apply to our graph.

#layouts <- layouts[!grepl("bipartite|merge|norm|sugiyama|tree", layouts)]


#par(mfrow=c(3,3), mar=c(1,1,1,1))

#for (layout in layouts) {
#  
#  print(layout)
#  
#  l <- do.call(layout, list(G_2)) 
#  
#  plot(G_2, edge.arrow.mode=0, layout=l, main=layout) }


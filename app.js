import express from "express";

import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import dotenv from "dotenv";
import fs from "fs/promises";

import { GoogleGenAI } from "@google/genai";
import { ChromaClient } from "chromadb";


app.set("view engine","ejs");
app.set("views",path.join(__dirname,"views"));

app.use(express.json());
app.use(express.urlencoded({ extended:true }));

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(express.json());

const ai = new GoogleGenAI();

const chroma=new ChromaClient({
    host:"localhost",
    port: "8000",
    ssl:false
});

const collection = await chroma.getOrCreateCollection({
    name:"knowledge"
}); 

async function createEmbeddings(text){
    const response = await ai.models.embedContent({
        model:"gemini-embedding-001",
        contents:[text],
    });

    return response.embeddings[0].values;
}

app.get("/clear",async(req,res)=>{
    try{
        const allItems=await collection.get();

        if(!allItems.ids || allItems.ids.length===0){
            return res.send("Collection is already Empty. Nothing to delete.");
        }

        await collection.delete({
            ids:allItems.ids
        });

        console.log(`Deleted ${allItems.ids.length} items from the chromaDB.🗑️`);
        res.send(`Succesfully deleted all ${allItems.ids.length} items from collection.`)
    }
    catch(Error){
        console.log("Error in deleting the componenets",Error);
        res.status(500).send("Failed to clear ChromaDB collection.");
    }
});


app.get("/ingest",async (req,res)=>{
    try{
    const text=await fs.readFile("./knowledge.txt","utf-8");

    const chunks = text
        .replace(/\r\n/g, "\n")  
        .split("\n\n")
        .map(chunk => chunk.trim())
        .filter(chunk => chunk.length > 0);
    
    for(let i=0;i<chunks.length;i++){
        const chunk=chunks[i];
        const embedding = await createEmbeddings(chunk);
        
        await collection.add({
            ids:[`chunk-${i}-${Date.now()}`],
            embeddings:[embedding],
            documents:[chunk]
        });
    }
    
    console.log("Stored in Chroma!✅");
    
    const allStoredItems=await collection.get({
        include:["embeddings","documents","metadatas"]
    });

    console.log("--- ALL STORED EMBEDDINGS IN CHROMA ---");
    res.send(JSON.stringify(allStoredItems,null,2));
    }

    catch(error){
        console.log("Cannot create embedings",error);
        res.status(500).send("Failed to create ChromaDB embeddings.");
    }
});

app.get("/getAnswer",async(req,res)=>{
    try{
    const question=`Where does photosynthesis takes place`;

    const questionEmbedding=await createEmbeddings(question);

    const result=await collection.query({
        queryEmbeddings:[questionEmbedding],
        nResults:2
    });

    const context=result.documents.join("\n\n");

    const prompt=`Answer the User's question using the provided context.
    
    
    Context: ${context}
    
    Question: ${question}`


    const response = await ai.models.generateContent({
        model:"gemini-2.5-flash",
        contents:prompt,
    });

    const answer = response.text;

    console.log(`---Answer has been generated successfully---✅`);
    res.send(answer);
}
catch(error){
    res.status(500).send({
        message:"Cannot generate a valid Response",
        error:error,
    });
}
});

app.listen(PORT, () => {
    console.log(`Server Running on ${PORT} ✅`);
});
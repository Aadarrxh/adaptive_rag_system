import express from "express";
import dotenv from "dotenv";
import fs from "fs/promises";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(express.json());

const ai = new GoogleGenAI();
const embeddedChunks = [];

async function createEmbedding(text) {
  const response = await ai.models.embedContent({
    model: "text-embedding-004", // Updated to the recommended embedding model
    contents: [text],
  });
  
  return response.embeddings[0].values;
}

function cosineSimilarity(vectorA, vectorB) {
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    magnitudeA += vectorA[i] * vectorA[i];
    magnitudeB += vectorB[i] * vectorB[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  return dotProduct / (magnitudeA * magnitudeB);
}

function searchSimilarChunks(questionEmbedding, embeddedChunks) {
  const results = embeddedChunks.map((chunk) => {
    const score = cosineSimilarity(questionEmbedding, chunk.embedding);
    return {
      text: chunk.text,
      score: score,
    };
  });

  results.sort((a, b) => b.score - a.score);
  return results;
}

app.get("/ingest", async (req, res) => {
  try {
    const text = await fs.readFile("./knowledge.txt", "utf-8");

    const chunks = text
      .replace(/\r\n/g, "\n")
      .split("\n\n")
      .map((chunk) => chunk.trim())
      .filter((chunk) => chunk.length > 0);

    // Reset array to avoid duplicate entries on re-ingestion
    embeddedChunks.length = 0;

    for (const chunk of chunks) {
      const embedding = await createEmbedding(chunk);

      embeddedChunks.push({
        text: chunk,
        embedding: embedding,
      });
    }

    console.log(`Successfully ingested ${embeddedChunks.length} chunks.`);
    res.send("Embeddings created successfully.");
  } catch (error) {
    console.error("Ingestion error:", error);
    res.status(500).send("Error creating embeddings.");
  }
});

// Answer endpoint: retrieves relevant chunks and queries the LLM
app.get("/getAnswer", async (req, res) => {
  try {
    if (embeddedChunks.length === 0) {
      return res.status(400).send("Please call /ingest first to populate knowledge base.");
    }

    const question = "Where does photosynthesis take place?";
    const questionEmbedding = await createEmbedding(question);

    console.log("Question:", question);
    console.log("Embedding length:", questionEmbedding.length);

    const result = searchSimilarChunks(questionEmbedding, embeddedChunks);
    const topChunks = result.slice(0, 2);

    const context = topChunks.map((chunk) => chunk.text).join("\n\n");

    const prompt = `Answer the user's question using the provided context.

Context:
${context}

Question: ${question}

Answer based on the context above.`;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
    });

    const answer = response.text;

    console.log("Retrieved chunks:", topChunks);
    console.log("Answer:", answer);

    res.send(answer);
  } catch (error) {
    console.error("Answer generation error:", error);
    res.status(500).send("Error processing request.");
  }
});

app.listen(PORT, () => {
  console.log(`Server Running on port ${PORT} ✅`);
});
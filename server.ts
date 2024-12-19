import 'dotenv/config'
import { ChatOpenAI } from "@langchain/openai";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { BufferMemory } from "langchain/memory";
import { 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    ChatPromptTemplate,
    AIMessagePromptTemplate 
} from "@langchain/core/prompts";
import { TwitterApi } from 'twitter-api-v2';
import OpenAI from 'openai';

const model = new ChatOpenAI({
  temperature: 0.7,
  modelName: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
  maxRetries: 3,
  maxConcurrency: 1,
});

async function testOpenAIDirectly() {
    const openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY
    });

    try {
        console.log("Testando chamada direta à API da OpenAI...");
        const completion = await openai.chat.completions.create({
            messages: [{ role: "user", content: "Diga olá!" }],
            model: "gpt-3.5-turbo",
        });

        console.log("Resposta:", completion.choices[0].message.content);
        return true;
    } catch (error) {
        console.error("Erro na chamada direta:", error);
        return false;
    }
}

await testOpenAIDirectly();

// Configuração do cliente Twitter
const twitterClient = new TwitterApi({
  appKey: 'asd',
  appSecret: 'asd',
  accessToken: 'asd-asd',
  accessSecret: 'asd',
});

// Interface for Binance API response
interface BinanceTickerResponse {
  symbol: string;
  lastPrice: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  highPrice: string;
  lowPrice: string;
  [key: string]: string | number; // for other fields we might not use
}

// Interface for our formatted data
interface FormattedCryptoData {
  symbol: string;
  lastPrice: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  highPrice: string;
  lowPrice: string;
}

const cryptoDataTool = new DynamicStructuredTool({
  name: "get_crypto_data",
  description: "Get current cryptocurrency price data from Binance",
  schema: {
      type: "object",
      properties: {
          symbol: {
              type: "string",
              description: "Trading pair symbol (e.g., BTCUSDT, ETHUSDT)"
          }
      },
      required: ["symbol"]
  },
  async func({ symbol }): Promise<string> {
      try {
          const url = `https://api.binance.com/api/v3/ticker/24hr?symbol=${symbol}`;
          console.log(`Buscando dados para ${symbol}...`);
          
          const response = await fetch(url);
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const data: BinanceTickerResponse = await response.json();
          
          // Format the data ensuring all values are strings
          const formattedData: FormattedCryptoData = {
              symbol: symbol,
              lastPrice: data.lastPrice,
              priceChangePercent: data.priceChangePercent,
              volume: data.volume,
              quoteVolume: data.quoteVolume,
              highPrice: data.highPrice,
              lowPrice: data.lowPrice
          };

          return JSON.stringify(formattedData, null, 2);
      } catch (error) {
          console.error(`Erro ao buscar dados para ${symbol}:`, error);
          throw error;
      }
  }
});

async function createAnalystAgent(memory) {
  const model = new ChatOpenAI({
      temperature: 0, // Reduced temperature to 0 for maximum precision
      modelName: "gpt-3.5-turbo",
      openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const systemMessage = `Você é um analista de criptomoedas que DEVE usar APENAS os dados exatos retornados pela API Binance.

  Ao receber os dados da API via get_crypto_data, você deve:
  
  1. Formatar os números adequadamente:
     - Preços (lastPrice, highPrice, lowPrice): Use $ com 2 casas decimais
     - Variação (priceChangePercent): Mantenha sinal e use 2 casas decimais
     - Volume: Mantenha 8 casas decimais para crypto, use formatação normal para USDT
  
  2. Use este formato exato para cada moeda:
  **[Symbol]**
  - Preço Atual: ${formatar o lastPrice como USD}
  - Variação 24h: ${formatar priceChangePercent}%
  - Volume: ${volume} {CRYPTO} / ${formatar quoteVolume como USD} USDT
  - Alta: ${formatar highPrice como USD}
  - Baixa: ${formatar lowPrice como USD}
  - Tendência: ${descrever com base na variação percentual}

  3. Para os tweets, use este formato:
  Tweet 1: Análise [SYMBOL]: Preço $[lastPrice], [priceChangePercent]% em 24h. Alta: $[highPrice], Baixa: $[lowPrice]
  Tweet 2: Volume [SYMBOL]: [volume] crypto / $[quoteVolume] USDT negociados em 24h
  Tweet 3: Resumo do mercado baseado apenas nos dados acima

  IMPORTANTE: Substitua todos os placeholders pelos valores reais formatados. Não deixe nenhum [placeholder] no texto final.`;

  const chatPrompt = ChatPromptTemplate.fromMessages([
      SystemMessagePromptTemplate.fromTemplate(systemMessage),
      new MessagesPlaceholder("chat_history"),
      HumanMessagePromptTemplate.fromTemplate("{input}"),
      AIMessagePromptTemplate.fromTemplate("{agent_scratchpad}")
  ]);

  const agent = await createOpenAIFunctionsAgent({
      llm: model,
      tools: [cryptoDataTool],
      prompt: chatPrompt
  });

  return new AgentExecutor({
      agent,
      tools: [cryptoDataTool],
      memory,
      verbose: true
  });
}

async function analyzeCrypto() {
  console.log("Iniciando análise do mercado...");
  
  const memory = new BufferMemory({
      returnMessages: true,
      memoryKey: "chat_history",
      outputKey: "output"
  });

  const analyst = await createAnalystAgent(memory);

  try {
      const analysis = await analyst.invoke({
          input: `Por favor, analise o mercado de criptomoedas:
                 1. Use get_crypto_data para obter dados do BTCUSDT
                 2. Use get_crypto_data para obter dados do ETHUSDT
                 3. Forneça uma análise com:
                    - Preços atuais
                    - Variações em 24h
                    - Volumes de negociação
                    - Tendências observadas
                 4. Crie um resumo em 3 tweets informativos`
      });

      console.log("\n=== Análise do Mercado Cripto ===");
      console.log(analysis.output);

      // // Preparar tweets
      // const tweets = analysis.output
      //     .split('\n')
      //     .filter(line => line.trim().length > 0)
      //     .map(line => line.trim())
      //     .filter(line => line.length <= 280);

      // // Postar no Twitter
      // if (tweets.length > 0) {
      //     console.log("\n=== Postando no Twitter ===");
      //     const posted = await postTwitterThread(tweets);
      //     if (posted) {
      //         console.log("Thread postada com sucesso!");
      //     } else {
      //         console.log("Erro ao postar thread.");
      //     }
      // }

      return analysis;
  } catch (error) {
      console.error("Erro na análise:", error);
      throw error;
  }
}

// Teste inicial da API
async function testBinanceAPI() {
  try {
      console.log("Testando API da Binance...");
      const response = await fetch('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT');
      if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log("Teste da API Binance OK - Preço BTC:", 
          parseFloat(data.lastPrice).toLocaleString('en-US', {
              style: 'currency',
              currency: 'USD'
          })
      );
      return true;
  } catch (error) {
      console.error("Erro no teste da API Binance:", error);
      return false;
  }
}

async function postTwitterThread(tweets) {
  console.log("Iniciando postagem no Twitter...");
  try {
      let lastTweetId = null;
      for (const tweet of tweets) {
          const tweetOptions = lastTweetId 
              ? { reply: { in_reply_to_tweet_id: lastTweetId } }
              : {};
          
          const response = await twitterClient.v2.tweet(tweet, tweetOptions);
          lastTweetId = response.data.id;
          console.log(`Tweet postado: ${tweet}`);
          
          // Espera 2 segundos entre tweets para evitar rate limits
          await new Promise(resolve => setTimeout(resolve, 2000));
      }
      console.log("Thread postada com sucesso!");
      return true;
  } catch (error) {
      console.error('Erro ao postar no Twitter:', error);
      return false;
  }
}

// Executar teste e análise
async function main() {
  const apiTest = await testBinanceAPI();
  if (apiTest) {
      await analyzeCrypto();
  } else {
      console.error("Falha no teste da API. Não prosseguindo com a análise.");
  }
}

main().catch(console.error);
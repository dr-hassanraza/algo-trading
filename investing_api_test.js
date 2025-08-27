#!/usr/bin/env node
/**
 * Investing.com API Test for PSX Stocks
 * =====================================
 * 
 * Tests the investing-com-api-v2 package with Pakistan Stock Exchange symbols
 * to see if we can get better data than EODHD API for fundamentals.
 */

const investing = require('investing-com-api-v2');

// PSX symbols to test
const psxSymbols = [
    'UBL',     // United Bank Limited
    'MCB',     // MCB Bank Limited
    'OGDC',    // Oil & Gas Development Company
    'LUCK',    // Lucky Cement
    'ENGRO',   // Engro Corporation
    'PPL',     // Pakistan Petroleum Limited
    'TRG',     // TRG Pakistan Limited
    'HBL',     // Habib Bank Limited
    'FFC',     // Fauji Fertilizer Company
    'NESTLE'   // Nestle Pakistan
];

async function testInvestingAPI() {
    console.log('🚀 Testing Investing.com API for PSX Stocks');
    console.log('=' .repeat(50));
    
    try {
        // Test 1: Search for PSX stocks
        console.log('\n📊 Test 1: Searching for PSX stocks...');
        
        for (let symbol of psxSymbols.slice(0, 3)) { // Test first 3 symbols
            try {
                console.log(`\n🔍 Searching for ${symbol}...`);
                
                // Search for the stock
                const searchResults = await investing.search(symbol + ' Pakistan');
                
                if (searchResults && searchResults.length > 0) {
                    console.log(`✅ Found ${searchResults.length} results for ${symbol}`);
                    
                    // Show first few results
                    searchResults.slice(0, 3).forEach((result, index) => {
                        console.log(`   ${index + 1}. ${result.name} (${result.symbol}) - ID: ${result.id}`);
                        console.log(`      Type: ${result.type} | Exchange: ${result.exchange || 'N/A'}`);
                    });
                    
                    // Try to get data for the first result
                    if (searchResults[0] && searchResults[0].id) {
                        try {
                            console.log(`\n📈 Getting data for ${searchResults[0].name}...`);
                            
                            // Get stock info
                            const stockInfo = await investing.getStockInfo(searchResults[0].id);
                            if (stockInfo) {
                                console.log(`   Current Price: ${stockInfo.price || 'N/A'}`);
                                console.log(`   Change: ${stockInfo.change || 'N/A'} (${stockInfo.changePercent || 'N/A'}%)`);
                                console.log(`   Volume: ${stockInfo.volume || 'N/A'}`);
                                console.log(`   Market Cap: ${stockInfo.marketCap || 'N/A'}`);
                                console.log(`   P/E Ratio: ${stockInfo.pe || 'N/A'}`);
                                console.log(`   EPS: ${stockInfo.eps || 'N/A'}`);
                            }
                            
                            // Get historical data
                            const historicalData = await investing.getHistoricalData(searchResults[0].id, '1M');
                            if (historicalData && historicalData.length > 0) {
                                console.log(`   Historical data: ${historicalData.length} data points`);
                                console.log(`   Latest: ${historicalData[0].date} - Close: ${historicalData[0].close}`);
                            }
                            
                        } catch (dataError) {
                            console.log(`   ⚠️ Could not get detailed data: ${dataError.message}`);
                        }
                    }
                    
                } else {
                    console.log(`❌ No results found for ${symbol}`);
                }
                
                // Add delay to avoid rate limiting
                await new Promise(resolve => setTimeout(resolve, 2000));
                
            } catch (symbolError) {
                console.log(`❌ Error searching for ${symbol}: ${symbolError.message}`);
            }
        }
        
        // Test 2: Try searching for Pakistan Stock Exchange directly
        console.log('\n📊 Test 2: Searching for "Pakistan Stock Exchange"...');
        try {
            const psxResults = await investing.search('Pakistan Stock Exchange');
            if (psxResults && psxResults.length > 0) {
                console.log(`✅ Found ${psxResults.length} PSX-related results`);
                psxResults.slice(0, 5).forEach((result, index) => {
                    console.log(`   ${index + 1}. ${result.name} - ${result.type}`);
                });
            }
        } catch (psxError) {
            console.log(`❌ PSX search error: ${psxError.message}`);
        }
        
        // Test 3: Try getting market indices
        console.log('\n📊 Test 3: Searching for KSE100 index...');
        try {
            const kseResults = await investing.search('KSE 100');
            if (kseResults && kseResults.length > 0) {
                console.log(`✅ Found ${kseResults.length} KSE100-related results`);
                kseResults.slice(0, 3).forEach((result, index) => {
                    console.log(`   ${index + 1}. ${result.name} (${result.symbol}) - ID: ${result.id}`);
                });
                
                // Try to get KSE100 data
                if (kseResults[0] && kseResults[0].id) {
                    try {
                        const kseData = await investing.getStockInfo(kseResults[0].id);
                        if (kseData) {
                            console.log(`   KSE100 Current: ${kseData.price || 'N/A'}`);
                            console.log(`   Change: ${kseData.change || 'N/A'} (${kseData.changePercent || 'N/A'}%)`);
                        }
                    } catch (kseError) {
                        console.log(`   ⚠️ Could not get KSE100 data: ${kseError.message}`);
                    }
                }
            } else {
                console.log(`❌ No KSE100 results found`);
            }
        } catch (kseError) {
            console.log(`❌ KSE search error: ${kseError.message}`);
        }
        
    } catch (error) {
        console.error(`❌ General API Error: ${error.message}`);
        console.error(`Stack: ${error.stack}`);
    }
    
    console.log('\n🏁 Test completed!');
}

// Run the test
if (require.main === module) {
    testInvestingAPI().catch(console.error);
}

module.exports = { testInvestingAPI };
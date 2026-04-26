#!/usr/bin/env node
/**
 * Explore Investing.com API Structure
 * ===================================
 * 
 * Let's explore what methods are available in the investing-com-api-v2 package
 */

const investing = require('investing-com-api-v2');

console.log('🔍 Exploring investing-com-api-v2 structure...');
console.log('=' .repeat(50));

console.log('\n📋 Available methods and properties:');
console.log('Type:', typeof investing);

if (typeof investing === 'object') {
    const methods = Object.getOwnPropertyNames(investing);
    console.log('Properties/Methods:', methods);
    
    methods.forEach(method => {
        console.log(`  ${method}: ${typeof investing[method]}`);
    });
} else if (typeof investing === 'function') {
    console.log('This appears to be a constructor function');
    console.log('Prototype methods:', Object.getOwnPropertyNames(investing.prototype || {}));
}

// Try different approaches
console.log('\n🧪 Testing different approaches...');

try {
    // Approach 1: Direct usage
    if (investing.search) {
        console.log('✅ Direct search method available');
    } else {
        console.log('❌ No direct search method');
    }
    
    // Approach 2: Constructor pattern
    if (typeof investing === 'function') {
        console.log('🔧 Trying constructor pattern...');
        const api = new investing();
        console.log('Constructor instance methods:', Object.getOwnPropertyNames(api));
    }
    
    // Approach 3: Check for default export
    if (investing.default) {
        console.log('🔧 Found default export, exploring...');
        const defaultApi = investing.default;
        console.log('Default export type:', typeof defaultApi);
        console.log('Default export methods:', Object.getOwnPropertyNames(defaultApi));
    }
    
} catch (error) {
    console.log('❌ Error exploring API:', error.message);
}

// Let's also check the package.json to understand the API better
try {
    const packagePath = './node_modules/investing-com-api-v2/package.json';
    const fs = require('fs');
    if (fs.existsSync(packagePath)) {
        const packageInfo = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
        console.log('\n📦 Package Info:');
        console.log('  Version:', packageInfo.version);
        console.log('  Description:', packageInfo.description);
        console.log('  Main:', packageInfo.main);
        console.log('  Homepage:', packageInfo.homepage);
    }
} catch (error) {
    console.log('❌ Could not read package info:', error.message);
}
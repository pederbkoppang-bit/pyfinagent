#!/usr/bin/env node
import { createServer } from 'https';
import { readFileSync } from 'fs';
import { spawn } from 'child_process';

const TLS_KEY = readFileSync('/Users/ford/.openclaw/proxy-key.pem');
const TLS_CERT = readFileSync('/Users/ford/.openclaw/proxy-cert.pem');

const PORT = parseInt(process.env.PROXY_PORT || '18796');
const CLAUDE_PATH = process.env.CLAUDE_PATH || 'claude';

// Concurrency control - only one claude -p at a time
let activeClaude = null; // Promise of current run, or null

const MODEL_MAP = {
  'claude-opus-4-6': 'opus', 'claude-opus-4-20250514': 'opus',
  'claude-opus-4-8': 'opus', 'claude-opus-4-7': 'opus',
  'claude-sonnet-4-6': 'sonnet', 'claude-sonnet-4-20250514': 'sonnet',
  'claude-sonnet-4-6-20250514': 'sonnet',
  'claude-sonnet-5': 'sonnet',
  'claude-fable-5': 'fable',
  'claude-haiku-4-5': 'haiku', 'claude-haiku-4-5-20251001': 'haiku',
};

// phase-76.9.2: unknown claude-* ids PASS THROUGH VERBATIM instead of the
// old silent-downgrade-to-sonnet trap -- the claude CLI accepts full model
// names, and a genuinely bad id now fails LOUDLY at the CLI instead of
// silently running a different model. Non-claude ids keep the sonnet default.
function resolveModel(m) {
  const id = (m||'').replace(/^anthropic\//,'');
  if (MODEL_MAP[id]) return MODEL_MAP[id];
  if (id.startsWith('claude-')) return id;
  return 'sonnet';
}

function extractPrompt(messages, system) {
  let parts = [];
  if (system) {
    const t = typeof system === 'string' ? system : (Array.isArray(system) ? system.map(b=>b.text||'').join('\n') : '');
    if (t) parts.push('[System]\n' + t);
  }
  for (const msg of messages || []) {
    const role = msg.role === 'assistant' ? 'Assistant' : 'User';
    const content = typeof msg.content === 'string' ? msg.content : (Array.isArray(msg.content) ? msg.content.map(b=>b.text||'').join('\n') : '');
    parts.push(`[${role}]\n${content}`);
  }
  return parts.join('\n\n');
}

const server = createServer({ key: TLS_KEY, cert: TLS_CERT }, (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }

  console.log(`[proxy] ${req.method} ${req.url}`);

  // Health
  if (req.url === '/health' || req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end('{"ok":true,"proxy":"claude-code-cli"}');
  }

  // OAuth usage - return unlimited
  if (req.url === '/api/oauth/usage' || req.url === '/v1/usage') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      rate_limit: {
        primary_window: { remaining: 999999, limit: 999999, reset_at: new Date(Date.now()+3600000).toISOString() },
        secondary_window: { remaining: 999999, limit: 999999, reset_at: new Date(Date.now()+60000).toISOString() },
      },
      usage: { input_tokens: 0, output_tokens: 0 },
      billing: { status: 'active', plan: 'max' },
    }));
  }

  // Models
  if (req.url === '/v1/models' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ object: 'list', data: Object.keys(MODEL_MAP).map(id => ({ id, object: 'model', owned_by: 'anthropic' })) }));
  }

  // Messages API
  if (req.url === '/v1/messages' && req.method === 'POST') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      let parsed;
      try { parsed = JSON.parse(body); } catch(e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ type: 'error', error: { type: 'parse_error', message: e.message } }));
      }

      const model = resolveModel(parsed.model);
      const prompt = extractPrompt(parsed.messages, parsed.system);
      const msgId = 'msg_' + Date.now();

      console.log(`[proxy] stream=${parsed.stream}, model=${parsed.model}, messages=${parsed.messages?.length}`);

      // Start SSE immediately
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'request-id': 'req_' + Date.now() });

      // Send message_start right away
      res.write(`event: message_start\ndata: ${JSON.stringify({
        type: 'message_start',
        message: { id: msgId, type: 'message', role: 'assistant', model: parsed.model,
          content: [], stop_reason: null, stop_sequence: null,
          usage: { input_tokens: Math.ceil(prompt.length/4), output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }
        }
      })}\n\n`);

      res.write(`event: content_block_start\ndata: ${JSON.stringify({
        type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' }
      })}\n\n`);

      // Ping every 2s
      const keepAlive = setInterval(() => {
        try { res.write(`event: ping\ndata: {"type":"ping"}\n\n`); } catch(e) {}
      }, 2000);

      // Serialize requests - wait for any active claude -p to finish
      const waitForPrev = activeClaude ? activeClaude.catch(() => {}) : Promise.resolve();

      let resolveActive;
      activeClaude = new Promise(r => { resolveActive = r; });

      waitForPrev.then(() => {
      // Run claude -p
      const args = ['-p', prompt, '--model', model, '--output-format', 'json', '--max-turns', '1'];
      console.log(`[proxy] Calling claude -p --model ${model} (${prompt.length} chars)`);
      const start = Date.now();
      const proc = spawn(CLAUDE_PATH, args, { env: { ...process.env, NO_COLOR: '1' }, timeout: 180000 });

      let stdout = '', stderr = '';
      proc.stdout.on('data', d => stdout += d.toString());
      proc.stderr.on('data', d => stderr += d.toString());

      proc.on('close', (code) => {
        resolveActive();
        clearInterval(keepAlive);
        console.log(`[proxy] claude -p finished in ${Date.now()-start}ms (code ${code})`);

        let text = '', output_tokens = 0;
        try {
          const r = JSON.parse(stdout);
          text = r.result || '';
          output_tokens = r.usage?.output_tokens || 0;
        } catch(e) {
          text = stdout.trim() || 'Error processing request';
        }

        // Send text in small chunks with tiny delays to mimic real streaming
        const chunks = [];
        for (let i = 0; i < text.length; i += 20) {
          chunks.push(text.slice(i, i + 20));
        }

        let idx = 0;
        const sendNext = () => {
          if (idx < chunks.length) {
            res.write(`event: content_block_delta\ndata: ${JSON.stringify({
              type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: chunks[idx] }
            })}\n\n`);
            idx++;
            setTimeout(sendNext, 10);  // 10ms between chunks
          } else {
            // All text sent, now close
            setTimeout(() => {
              res.write(`event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n`);
              setTimeout(() => {
                res.write(`event: message_delta\ndata: ${JSON.stringify({ type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens } })}\n\n`);
                setTimeout(() => {
                  res.write(`event: message_stop\ndata: {"type":"message_stop"}\n\n`);
                  res.end();
                }, 10);
              }, 10);
            }, 10);
          }
        };
        sendNext();
      });

      proc.on('error', (err) => {
        resolveActive();
        clearInterval(keepAlive);
        console.error('[proxy error]', err.message);
        res.write(`event: error\ndata: ${JSON.stringify({ type: 'error', error: { type: 'proxy_error', message: err.message } })}\n\n`);
        res.end();
      });
      }); // end waitForPrev.then
    });
    return;
  }

  // Catch-all
  console.log(`[proxy] UNHANDLED: ${req.method} ${req.url}`);
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end('{"error":"not found"}');
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[claude-code-proxy] Listening on https://127.0.0.1:${PORT}`);
});

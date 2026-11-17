import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App.jsx';

// Mock fetch globally
global.fetch = vi.fn();

describe('App Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock successful API responses
    global.fetch.mockImplementation((url) => {
      if (url.includes('/api/health')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ status: 'healthy', model: 'best.pt' })
        });
      }
      if (url.includes('/api/checkpoints')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ checkpoints: ['best.pt', 'final.pt', 'latest.pt'] })
        });
      }
      if (url.includes('/api/status')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ 
            status: 'ready', 
            model: 'best.pt',
            device: 'cuda:0'
          })
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({})
      });
    });
  });

  it('renders without crashing', () => {
    render(<App />);
    expect(screen.getByText(/Ragnarok AI/i)).toBeInTheDocument();
  });

  it('displays generation parameters', () => {
    render(<App />);
    expect(screen.getByLabelText(/temperature/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/top-k/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/top-p/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/max tokens/i)).toBeInTheDocument();
  });

  it('shows generate button', () => {
    render(<App />);
    expect(screen.getByRole('button', { name: /generate/i })).toBeInTheDocument();
  });
});

describe('SSE Parser', () => {
  // Import the actual SSE parser
  const { createSSEParser } = await import('../sse.js');

  it('parses start event', () => {
    const events = [];
    const feed = createSSEParser((event, data) => events.push([event, data]));
    
    feed('event: start\ndata: {"step": 1}\n\n');
    
    expect(events).toHaveLength(1);
    expect(events[0][0]).toBe('start');
    expect(events[0][1]).toEqual({ step: 1 });
  });

  it('parses token events', () => {
    const events = [];
    const feed = createSSEParser((event, data) => events.push([event, data]));
    
    feed('event: token\ndata: {"text": "Hello"}\n\n');
    
    expect(events).toHaveLength(1);
    expect(events[0][0]).toBe('token');
    expect(events[0][1]).toEqual({ text: 'Hello' });
  });

  it('handles multiple events in one chunk', () => {
    const events = [];
    const feed = createSSEParser((event, data) => events.push([event, data]));
    
    feed('event: start\ndata: {"step": 1}\n\nevent: token\ndata: {"text": "Hi"}\n\n');
    
    expect(events).toHaveLength(2);
    expect(events[0][0]).toBe('start');
    expect(events[1][0]).toBe('token');
  });

  it('handles split chunks', () => {
    const events = [];
    const feed = createSSEParser((event, data) => events.push([event, data]));
    
    const raw = 'event: token\ndata: {"text": "hello"}\n\n';
    for (let i = 0; i < raw.length; i += 3) {
      feed(raw.slice(i, i + 3));
    }
    
    expect(events).toHaveLength(1);
    expect(events[0][1].text).toBe('hello');
  });
});

describe('API Service', () => {
  const { generateText, listCheckpoints, loadCheckpoint, getStatus } = await import('../api.js');

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('generates text via SSE', async () => {
    // Mock SSE response
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode('event: start\ndata: {"step": 1}\n\n'));
        controller.enqueue(encoder.encode('event: token\ndata: {"text": "Hello"}\n\n'));
        controller.enqueue(encoder.encode('event: done\ndata: {"tokens": 1}\n\n'));
        controller.close();
      }
    });
    
    global.fetch.mockResolvedValue({
      ok: true,
      body: stream
    });

    const tokens = [];
    await generateText(
      'Test prompt',
      { temperature: 0.7, top_k: 40, top_p: 0.9, max_tokens: 100 },
      (token) => tokens.push(token)
    );

    expect(tokens).toContain('Hello');
  });

  it('lists checkpoints', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ 
        checkpoints: ['best.pt', 'final.pt'] 
      })
    });

    const result = await listCheckpoints();
    expect(result.checkpoints).toEqual(['best.pt', 'final.pt']);
  });

  it('loads checkpoint', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ success: true })
    });

    const result = await loadCheckpoint('best.pt');
    expect(result.success).toBe(true);
  });
});
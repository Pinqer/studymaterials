/**
 * Universal Search Module
 * Full-text search across all content including phrase matching
 */

class UniversalSearch {
    constructor() {
        this.searchIndex = [];
        this.fuseInstance = null;
        this.loaded = false;
    }

    /**
     * Load the full search index
     */
    async loadSearchIndex() {
        try {
            const response = await fetch('data/search_index.json');
            const data = await response.json();
            this.searchIndex = data.items || [];

            // Initialize Fuse.js for fuzzy matching
            this.fuseInstance = new Fuse(this.searchIndex, {
                includeScore: true,
                includeMatches: true,
                threshold: 0.3,
                ignoreLocation: true,
                minMatchCharLength: 2,
                keys: [
                    { name: 'title', weight: 2 },
                    { name: 'content', weight: 1 }
                ]
            });

            this.loaded = true;
            console.log(`Search index loaded: ${this.searchIndex.length} items`);
        } catch (error) {
            console.warn('Could not load search_index.json:', error);
        }
    }

    /**
     * Universal search - uses phrase matching for long queries, fuzzy for short
     */
    search(query) {
        if (!query || query.length < 2) return [];

        const words = query.trim().split(/\s+/);

        // For long queries (4+ words), use exact phrase matching
        if (words.length >= 4) {
            return this.phraseSearch(query);
        }

        // For short queries, use a combination of fuzzy + phrase
        const fuzzyResults = this.fuzzySearch(query);
        const phraseResults = this.phraseSearch(query);

        // Combine and deduplicate
        const combined = this.mergeResults(phraseResults, fuzzyResults);
        return combined.slice(0, 15);
    }

    /**
     * Exact phrase/substring search
     */
    phraseSearch(query) {
        const queryLower = query.toLowerCase();
        const results = [];

        for (const item of this.searchIndex) {
            const content = (item.content || '').toLowerCase();
            const title = (item.title || '').toLowerCase();

            let found = false;
            let matchIndex = -1;

            // Check title first
            if (title.includes(queryLower)) {
                found = true;
                matchIndex = title.indexOf(queryLower);
            }
            // Then check content
            else if (content.includes(queryLower)) {
                found = true;
                matchIndex = content.indexOf(queryLower);
            }

            if (found) {
                // Extract excerpt around match
                const contextStart = Math.max(0, matchIndex - 50);
                const contextEnd = Math.min(content.length, matchIndex + query.length + 100);
                let excerpt = content.substring(contextStart, contextEnd);
                if (contextStart > 0) excerpt = '...' + excerpt;
                if (contextEnd < content.length) excerpt = excerpt + '...';

                results.push({
                    ...item,
                    score: 0.1, // High priority for exact matches
                    excerpt: excerpt
                });
            }
        }

        return results;
    }

    /**
     * Fuzzy search using Fuse.js
     */
    fuzzySearch(query) {
        if (!this.fuseInstance) return [];

        const results = this.fuseInstance.search(query);
        return results
            .filter(r => r.score < 0.5)
            .slice(0, 20)
            .map(r => ({
                ...r.item,
                score: r.score,
                matches: r.matches
            }));
    }

    /**
     * Merge and deduplicate results
     */
    mergeResults(primary, secondary) {
        const seen = new Set();
        const merged = [];

        // Add primary results first (phrase matches)
        for (const item of primary) {
            const key = item.id + item.source;
            if (!seen.has(key)) {
                seen.add(key);
                merged.push(item);
            }
        }

        // Add secondary results
        for (const item of secondary) {
            const key = item.id + item.source;
            if (!seen.has(key)) {
                seen.add(key);
                merged.push(item);
            }
        }

        return merged;
    }

    /**
     * Get excerpt around matched text
     */
    getExcerpt(text, query, maxLength = 150) {
        if (!text) return '';

        const queryLower = query.toLowerCase();
        const textLower = text.toLowerCase();
        const index = textLower.indexOf(queryLower);

        if (index === -1) {
            return text.substring(0, maxLength) + (text.length > maxLength ? '...' : '');
        }

        const start = Math.max(0, index - 50);
        const end = Math.min(text.length, index + query.length + 100);
        let excerpt = text.substring(start, end);

        if (start > 0) excerpt = '...' + excerpt;
        if (end < text.length) excerpt = excerpt + '...';

        return excerpt;
    }

    /**
     * Highlight query in text
     */
    highlightMatch(text, query) {
        if (!text || !query) return text;

        const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedQuery})`, 'gi');
        return text.replace(regex, '<span class="highlight">$1</span>');
    }
}

// Legacy FuzzySearch class for backward compatibility
class FuzzySearch {
    constructor() {
        this.fuseSlides = null;
        this.fuseNotes = null;
        this.fuseExercises = null;
        this.fuseExams = null;
        this.fuseAll = null;
        this.universalSearch = new UniversalSearch();
    }

    initialize(slides, notes, exercises, exams = [], studyGuide = []) {
        const baseOptions = {
            includeScore: true,
            includeMatches: true,
            threshold: 0.2,
            ignoreLocation: true,
            minMatchCharLength: 3,
            distance: 100
        };

        this.fuseSlides = new Fuse(slides, {
            ...baseOptions,
            keys: [
                { name: 'title', weight: 2 },
                { name: 'topic', weight: 1.5 },
                { name: 'description', weight: 1 },
                { name: 'keywords', weight: 1.5 }
            ]
        });

        this.fuseNotes = new Fuse(notes, {
            ...baseOptions,
            keys: [
                { name: 'title', weight: 2 },
                { name: 'topic', weight: 1.5 },
                { name: 'content', weight: 1 },
                { name: 'keywords', weight: 1.5 }
            ]
        });

        this.fuseExercises = new Fuse(exercises, {
            ...baseOptions,
            keys: [
                { name: 'title', weight: 2 },
                { name: 'topic', weight: 1.5 },
                { name: 'question', weight: 1.5 },
                { name: 'answer', weight: 1 },
                { name: 'keywords', weight: 1.5 }
            ]
        });

        this.fuseExams = new Fuse(exams, {
            ...baseOptions,
            keys: [
                { name: 'title', weight: 2 },
                { name: 'topic', weight: 1.5 },
                { name: 'description', weight: 1 },
                { name: 'keywords', weight: 1.5 },
                { name: 'year', weight: 1 }
            ]
        });

        // Load universal search index
        this.universalSearch.loadSearchIndex();
    }

    /**
     * Search using universal search (full-text + fuzzy)
     */
    searchAll(query) {
        if (!query) return [];

        // Use universal search if available
        if (this.universalSearch.loaded) {
            return this.universalSearch.search(query);
        }

        // Fallback to basic Fuse search
        return this._legacySearchAll(query);
    }

    _legacySearchAll(query) {
        if (!this.fuseAll) return [];
        const results = this.fuseAll.search(query);
        return results
            .filter(result => result.score < 0.4)
            .slice(0, 10)
            .map(result => ({
                ...result.item,
                score: result.score,
                matches: result.matches
            }));
    }

    searchSlides(query) {
        if (!query || !this.fuseSlides) return [];
        const results = this.fuseSlides.search(query);
        return results
            .filter(result => result.score < 0.4)
            .map(result => ({
                ...result.item,
                score: result.score,
                matches: result.matches
            }));
    }

    searchNotes(query) {
        if (!query || !this.fuseNotes) return [];
        const results = this.fuseNotes.search(query);
        return results
            .filter(result => result.score < 0.4)
            .map(result => ({
                ...result.item,
                score: result.score,
                matches: result.matches
            }));
    }

    searchExercises(query) {
        if (!query || !this.fuseExercises) return [];
        const results = this.fuseExercises.search(query);
        return results
            .filter(result => result.score < 0.4)
            .map(result => ({
                ...result.item,
                score: result.score,
                matches: result.matches
            }));
    }

    searchExams(query) {
        if (!query || !this.fuseExams) return [];
        const results = this.fuseExams.search(query);
        return results
            .filter(result => result.score < 0.4)
            .map(result => ({
                ...result.item,
                score: result.score,
                matches: result.matches
            }));
    }

    highlightMatches(text, matches, key) {
        if (!matches || !text) return this.escapeHtml(text);

        const relevantMatches = matches.filter(m => m.key === key);
        if (relevantMatches.length === 0) return this.escapeHtml(text);

        let result = text;
        const indices = [];

        relevantMatches.forEach(match => {
            match.indices.forEach(([start, end]) => {
                indices.push({ start, end: end + 1 });
            });
        });

        indices.sort((a, b) => b.start - a.start);

        indices.forEach(({ start, end }) => {
            const before = result.substring(0, start);
            const matchText = result.substring(start, end);
            const after = result.substring(end);
            result = `${before}__HIGHLIGHT_START__${matchText}__HIGHLIGHT_END__${after}`;
        });

        result = this.escapeHtml(result);
        result = result.replace(/__HIGHLIGHT_START__/g, '<span class="highlight">');
        result = result.replace(/__HIGHLIGHT_END__/g, '</span>');

        return result;
    }

    escapeHtml(text) {
        if (!text) return '';
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    getExcerpt(text, matches, key, maxLength = 150) {
        if (!text) return '';

        let cleanText = text
            .replace(/\$\$[\s\S]*?\$\$/g, '[formula]')
            .replace(/\$[^\$\n]+?\$/g, '[formula]')
            .replace(/\\[a-zA-Z]+/g, '')
            .replace(/[{}]/g, '')
            .replace(/\*\*/g, '')
            .replace(/\*/g, '')
            .replace(/###?\s*/g, '')
            .replace(/\n+/g, ' ')
            .trim();

        const relevantMatches = matches?.filter(m => m.key === key);

        if (!relevantMatches || relevantMatches.length === 0) {
            const excerpt = cleanText.substring(0, maxLength);
            return this.escapeHtml(excerpt) + (cleanText.length > maxLength ? '...' : '');
        }

        const firstMatch = relevantMatches[0].indices[0];
        const matchStart = firstMatch[0];
        const matchEnd = firstMatch[1] + 1;
        const matchedText = text.substring(matchStart, matchEnd);

        const excerptStart = Math.max(0, matchStart - 50);
        const excerptEnd = Math.min(text.length, matchStart + maxLength - 50);

        let excerpt = text.substring(excerptStart, excerptEnd);

        excerpt = excerpt
            .replace(/\$\$[\s\S]*?\$\$/g, '[formula]')
            .replace(/\$[^\$\n]+?\$/g, '[formula]')
            .replace(/\\[a-zA-Z]+/g, '')
            .replace(/[{}]/g, '')
            .replace(/\*\*/g, '')
            .replace(/\*/g, '')
            .replace(/###?\s*/g, '')
            .replace(/\n+/g, ' ')
            .trim();

        if (excerptStart > 0) excerpt = '...' + excerpt;
        if (excerptEnd < text.length) excerpt = excerpt + '...';

        excerpt = this.escapeHtml(excerpt);

        const escapedMatch = this.escapeHtml(matchedText).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedMatch})`, 'gi');
        excerpt = excerpt.replace(regex, '<span class="highlight">$1</span>');

        return excerpt;
    }
}

// Global instances
const fuzzySearch = new FuzzySearch();
const universalSearch = new UniversalSearch();

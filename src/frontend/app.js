const { useEffect, useMemo, useRef, useState } = React;

const INTENT_LABELS = {
    similar_items: "Similar Picks",
    color_variant: "Color Variants",
    graph_pairing: "Outfit Pairing",
};

const DEFAULT_BOOTSTRAP = {
    app: {
        name: "Atelier mRAG",
        tagline: "AI Fashion Styling Studio",
        api_version: "2",
        theme: "retro-vintage",
    },
    suggested_prompts: [
        "Find pants that match this top",
        "Show color alternatives for #article_id",
        "Build a smart-casual outfit from this item",
        "Find a cleaner minimal version of this look",
    ],
};

function uid() {
    if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    return "id-" + Date.now() + "-" + Math.random().toString(16).slice(2);
}

function getSessionId() {
    const key = "mrag_session_id";
    let sid = localStorage.getItem(key);
    if (!sid) {
        sid = uid();
        localStorage.setItem(key, sid);
    }
    return sid;
}

function formatArticleId(value) {
    const str = String(value || "").trim();
    return /^\d+$/.test(str) ? str.padStart(10, "0") : str;
}

function imagePathFromId(articleId) {
    const id = formatArticleId(articleId);
    if (!id) return "";
    return "/images/" + id.substring(0, 3) + "/" + id + ".jpg";
}

function normalizeItems(items) {
    return (items || [])
        .map((item) => ({ ...item, article_id: formatArticleId(item.article_id) }))
        .filter((item) => item.article_id);
}

function ProductModal({ item, onClose }) {
    if (!item) return null;

    const aid = formatArticleId(item.article_id);
    const title = item.title || item.product_type || "Product";
    const subtitle = item.subtitle || item.colour_group || "-";

    return (
        <div id="product-modal" onClick={onClose}>
            <div className="panel rounded-2xl w-full max-w-4xl p-6 md:p-8 grid grid-cols-1 md:grid-cols-[330px_1fr] gap-6 relative" onClick={(e) => e.stopPropagation()}>
                <button className="absolute top-4 right-4 text-2xl text-white/60 hover:text-white" onClick={onClose}>&times;</button>
                <img className="w-full h-80 object-cover rounded-xl border border-white/10" src={item.image_url || imagePathFromId(aid)} alt="Product" />
                <div>
                    <h3 className="font-display text-4xl leading-tight mb-1">{title}</h3>
                    <p className="text-sm text-[#b5a58c] mb-4 font-code">#{aid}</p>
                    <div className="grid sm:grid-cols-2 gap-3 text-sm">
                        <div><span className="text-[#b5a58c] uppercase text-xs">Color</span><p>{subtitle}</p></div>
                        <div><span className="text-[#b5a58c] uppercase text-xs">Fit</span><p>{item.fit || "-"}</p></div>
                        <div><span className="text-[#b5a58c] uppercase text-xs">Occasion</span><p>{item.occasion || "-"}</p></div>
                        <div><span className="text-[#b5a58c] uppercase text-xs">Season</span><p>{item.seasonality || "-"}</p></div>
                    </div>
                    <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-4">
                        <p className="text-sm leading-relaxed text-[#dfd2bd]">{item.description || "-"}</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

function HeroScreen({ bootstrap, onStart }) {
    const appName = (bootstrap && bootstrap.app && bootstrap.app.name) || DEFAULT_BOOTSTRAP.app.name;
    const tagline = (bootstrap && bootstrap.app && bootstrap.app.tagline) || DEFAULT_BOOTSTRAP.app.tagline;
    const rootRef = useRef(null);

    useEffect(() => {
        if (!window.gsap || !rootRef.current) return undefined;

        const q = window.gsap.utils.selector(rootRef);
        const tl = window.gsap.timeline({ defaults: { ease: "power3.out" } });

        tl.from(q(".hero-badge"), { y: 14, opacity: 0, duration: 0.45 })
            .from(q(".hero-title"), { y: 24, opacity: 0, duration: 0.65 }, "-=0.16")
            .from(q(".hero-sub"), { y: 16, opacity: 0, duration: 0.52 }, "-=0.28")
            .from(q(".hero-enter-btn"), { y: 8, opacity: 0, duration: 0.36 }, "-=0.12");

        return () => tl.kill();
    }, []);

    return (
        <section className="hero" ref={rootRef}>
            <div className="max-w-5xl">
                <p className="retro-badge hero-badge mx-auto mb-5">{tagline}</p>
                <h1 className="hero-title">{appName}<br />Styling Console</h1>
                <p className="hero-sub">
                    Upload a reference image, refine your request, and explore outfit suggestions in a clean visual flow.
                </p>
                <button onClick={onStart} className="btn hero-enter-btn mt-9 px-11 py-4 text-lg font-semibold">
                    Start Styling
                </button>
            </div>
        </section>
    );
}

function StatusRibbon({ meta, sessionId }) {
    if (!meta) {
        return (
            <div className="status-ribbon">
                <span className="status-pill">Session {sessionId.slice(0, 8)}</span>
                <span className="status-pill">Ready</span>
            </div>
        );
    }

    return (
        <div className="status-ribbon">
            <span className="status-pill">Session {sessionId.slice(0, 8)}</span>
            <span className="status-pill">Latency {meta.latency_ms || 0}ms</span>
            <span className="status-pill">Results {meta.result_count || 0}</span>
            <span className="status-pill">Req {String(meta.request_id || "-").slice(0, 8)}</span>
        </div>
    );
}

function UserMessage({ message }) {
    return (
        <div className="chat-message flex flex-col items-end gap-2">
            {message.image && <img src={message.image} className="max-w-[220px] md:max-w-[260px] rounded-xl border border-white/20" alt="Uploaded" />}
            {message.text && <div className="msg-user">{message.text}</div>}
        </div>
    );
}

function ItemCard({ item, onOpen }) {
    const cardRef = useRef(null);

    function onMouseMove(event) {
        const el = cardRef.current;
        if (!el) return;

        const rect = el.getBoundingClientRect();
        const px = (event.clientX - rect.left) / rect.width;
        const py = (event.clientY - rect.top) / rect.height;
        const rx = (0.5 - py) * 8;
        const ry = (px - 0.5) * 9;

        el.style.transform = "perspective(860px) rotateX(" + rx.toFixed(2) + "deg) rotateY(" + ry.toFixed(2) + "deg) translateY(-2px)";
    }

    function onMouseLeave() {
        const el = cardRef.current;
        if (!el) return;
        el.style.transform = "";
    }

    const aid = formatArticleId(item.article_id);
    const title = item.title || item.product_type || "Item";
    const subtitle = item.subtitle || item.colour_group || "";

    return (
        <div
            ref={cardRef}
            className={"item-card " + (item.is_anchor ? "anchor" : "")}
            onMouseMove={onMouseMove}
            onMouseLeave={onMouseLeave}
            onClick={onOpen}
        >
            <div className="overflow-hidden rounded-md border border-white/10 bg-black/25 mb-3">
                <img src={item.image_url || imagePathFromId(aid)} className="w-full h-44 object-cover" alt="Product" />
            </div>
            <div className="flex items-center justify-between gap-2">
                <p className="text-[11px] text-[#baa98d] font-code">{aid}</p>
                {item.is_anchor && <span className="text-[10px] uppercase tracking-[0.12em] text-[#ddb87f]">focus item</span>}
            </div>
            <p className="font-display text-3xl leading-[0.9] mt-1">{title}</p>
            <p className="text-xs text-[#baa98d] mt-1">{subtitle}</p>
        </div>
    );
}

function AiMessage({ message, onOpenItem }) {
    const intentLabel = (message.intentInfo && message.intentInfo.label) || INTENT_LABELS[message.intent] || "";
    const intentDescription = (message.intentInfo && message.intentInfo.description) || "";
    const cards = Array.isArray(message.cards) && message.cards.length > 0 ? message.cards : (message.items || []);

    return (
        <div className="chat-message flex flex-col items-start gap-3">
            <div className={"msg-ai " + (message.error ? "text-red-300" : "")}>{message.text}</div>
            {intentLabel && <span className="intent-chip">{intentLabel}</span>}
            {intentDescription && <p className="text-xs text-[#baa98d] -mt-1">{intentDescription}</p>}

            {Array.isArray(cards) && cards.length > 0 && (
                <div className="cards-row w-full">
                    {cards.map((item, index) => (
                        <ItemCard key={String(item.article_id) + "-" + index} item={item} onOpen={() => onOpenItem(item)} />
                    ))}
                </div>
            )}
        </div>
    );
}

function QuickActions({ actions, onPick }) {
    if (!Array.isArray(actions) || actions.length === 0) return null;

    return (
        <div className="quick-actions">
            {actions.map((action, index) => (
                <button key={action + "-" + index} className="quick-action-chip" onClick={() => onPick(action)}>
                    {action}
                </button>
            ))}
        </div>
    );
}

function ReferencePicker({ items, selectedAnchorId, onSelectAnchor }) {
    return (
        <div className="panel p-3">
            <p className="retro-badge w-fit mb-2">Choose Focus Item</p>

            {!items.length && (
                <div className="text-sm text-[#a89372] rounded-xl border border-white/10 bg-black/20 p-3">
                    Once suggestions appear, choose a focus item for your next message.
                </div>
            )}

            {items.length > 0 && (
                <div className="reference-list">
                    {items.map((item) => {
                        const aid = formatArticleId(item.article_id);
                        const isActive = aid === selectedAnchorId;
                        const title = item.product_type || item.title || "Item";
                        const subtitle = item.colour_group || item.subtitle || "";

                        return (
                            <button
                                key={aid}
                                className={"reference-item " + (isActive ? "active" : "")}
                                onClick={() => onSelectAnchor(aid)}
                            >
                                <img
                                    src={item.image_url || imagePathFromId(aid)}
                                    alt={title}
                                    className="reference-thumb"
                                />
                                <div className="reference-meta">
                                    <p className="title">{title}</p>
                                    <p className="subtitle">{subtitle || "No color specified"}</p>
                                    <p className="article">#{aid}</p>
                                </div>
                                <span className="reference-cta">Use</span>
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

function MoodBoardPanel() {
    return (
        <div className="panel p-3 mood-board">
            <p className="retro-badge w-fit mb-2">Style Mood</p>
            <div className="mood-grid">
                <img src="https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=480&q=80" alt="Style mood 1" />
                <img src="https://images.unsplash.com/photo-1529139574466-a303027c1d8b?auto=format&fit=crop&w=480&q=80" alt="Style mood 2" />
            </div>
            <p className="text-xs text-[#c8b497] mt-2 leading-relaxed">
                Tip: select one focus item first, then ask for matching bottoms, outerwear, or color variants.
            </p>
        </div>
    );
}

function Composer({
    chatInput,
    setChatInput,
    currentImageBase64,
    onUploadClick,
    onFileChange,
    onRemoveImage,
    onSend,
    fileInputRef,
    textAreaRef,
    isLoading,
}) {
    return (
        <div className="composer">
            {currentImageBase64 && (
                <div className="flex items-center gap-3">
                    <img src={currentImageBase64} className="h-20 w-20 object-cover rounded border border-white/20" alt="Preview" />
                    <div>
                        <p className="text-sm text-[#cebda1]">A new image resets the context and focus item.</p>
                        <button className="text-sm text-red-300 hover:text-red-200 mt-1" onClick={onRemoveImage}>Remove image</button>
                    </div>
                </div>
            )}

            <div className="composer-row">
                <input ref={fileInputRef} type="file" className="hidden" accept="image/*" onChange={onFileChange} />
                <button className="btn h-[52px] w-[52px] flex items-center justify-center" disabled={isLoading} onClick={onUploadClick} aria-label="Upload image">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                </button>
                <textarea
                    ref={textAreaRef}
                    className="composer-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Example: Find shoes that match this shirt"
                    onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            onSend();
                        }
                    }}
                />
                <button className="btn h-[52px] px-7 font-semibold" disabled={isLoading} onClick={onSend}>
                    {isLoading ? "Sending..." : "Send"}
                </button>
            </div>
        </div>
    );
}

function Workspace({
    sessionId,
    messages,
    quickActions,
    selectedAnchorId,
    anchorItems,
    onSelectAnchor,
    onOpenItem,
    onPickQuickAction,
    onResetSession,
    onSend,
    chatInput,
    setChatInput,
    currentImageBase64,
    onUploadClick,
    onFileChange,
    onRemoveImage,
    fileInputRef,
    textAreaRef,
    isLoading,
    chatBoxRef,
    latestMeta,
}) {
    return (
        <section className="workspace">
            <div className="layout">
                <div className="panel chat-panel">
                    <StatusRibbon meta={latestMeta} sessionId={sessionId} />

                    <div className="chat-header">
                        <div>
                            <p className="text-xs uppercase tracking-[0.2em] text-[#c8b69a]">Styling Session</p>
                            <h2 className="font-display text-3xl">Conversation Canvas</h2>
                        </div>
                        <div className="flex items-center gap-2">
                            <button className="btn px-3 py-2 text-xs" onClick={onResetSession}>Reset Session</button>
                            <span className={"anchor-pill " + (!selectedAnchorId ? "hidden" : "")}>Focus item <strong>#{selectedAnchorId}</strong></span>
                        </div>
                    </div>

                    <QuickActions actions={quickActions} onPick={onPickQuickAction} />

                    <div className="chat-box" ref={chatBoxRef}>
                        {messages.map((message) => (
                            message.role === "user"
                                ? <UserMessage key={message.id} message={message} />
                                : <AiMessage key={message.id} message={message} onOpenItem={onOpenItem} />
                        ))}
                        {isLoading && <div className="typing">Generating outfit suggestions...</div>}
                    </div>

                    <Composer
                        chatInput={chatInput}
                        setChatInput={setChatInput}
                        currentImageBase64={currentImageBase64}
                        onUploadClick={onUploadClick}
                        onFileChange={onFileChange}
                        onRemoveImage={onRemoveImage}
                        onSend={onSend}
                        fileInputRef={fileInputRef}
                        textAreaRef={textAreaRef}
                        isLoading={isLoading}
                    />
                </div>

                <div className="side-panel">
                    <ReferencePicker items={anchorItems} selectedAnchorId={selectedAnchorId} onSelectAnchor={onSelectAnchor} />
                    <MoodBoardPanel />
                </div>
            </div>
        </section>
    );
}

function App() {
    const [started, setStarted] = useState(false);
    const [bootstrap, setBootstrap] = useState(DEFAULT_BOOTSTRAP);
    const [messages, setMessages] = useState([]);
    const [chatInput, setChatInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [currentImageBase64, setCurrentImageBase64] = useState(null);
    const [selectedAnchorId, setSelectedAnchorId] = useState("");
    const [anchorItems, setAnchorItems] = useState([]);
    const [quickActions, setQuickActions] = useState(DEFAULT_BOOTSTRAP.suggested_prompts.slice(0, 6));
    const [modalItem, setModalItem] = useState(null);
    const [latestMeta, setLatestMeta] = useState(null);

    const chatBoxRef = useRef(null);
    const fileInputRef = useRef(null);
    const textAreaRef = useRef(null);
    const sessionIdRef = useRef(getSessionId());

    const api = useMemo(() => window.MragApi || {}, []);

    useEffect(() => {
        let alive = true;
        const loader = api.getBootstrap
            ? api.getBootstrap()
            : fetch("/api/frontend/bootstrap").then((response) => response.json());

        Promise.resolve(loader)
            .then((data) => {
                if (!alive || !data) return;
                setBootstrap((prev) => ({ ...prev, ...data }));
                if (Array.isArray(data.suggested_prompts) && data.suggested_prompts.length > 0) {
                    setQuickActions(data.suggested_prompts.slice(0, 6));
                }
            })
            .catch(() => {
                // Keep default bootstrap values.
            });

        return () => {
            alive = false;
        };
    }, [api]);

    useEffect(() => {
        if (!started) return undefined;

        const html = document.documentElement;
        const body = document.body;
        const prevHtmlOverflow = html.style.overflow;
        const prevBodyOverflow = body.style.overflow;
        const prevHtmlHeight = html.style.height;
        const prevBodyHeight = body.style.height;

        html.style.overflow = "hidden";
        body.style.overflow = "hidden";
        html.style.height = "100%";
        body.style.height = "100%";
        window.scrollTo(0, 0);

        return () => {
            html.style.overflow = prevHtmlOverflow;
            body.style.overflow = prevBodyOverflow;
            html.style.height = prevHtmlHeight;
            body.style.height = prevBodyHeight;
        };
    }, [started]);

    useEffect(() => {
        if (!chatBoxRef.current) return;
        chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }, [messages, isLoading]);

    useEffect(() => {
        if (!window.gsap || !chatBoxRef.current || messages.length === 0) return;
        const nodes = chatBoxRef.current.querySelectorAll(".chat-message");
        const lastNode = nodes[nodes.length - 1];
        if (!lastNode) return;
        window.gsap.fromTo(
            lastNode,
            { opacity: 0, y: 14 },
            { opacity: 1, y: 0, duration: 0.34, ease: "power2.out" }
        );
    }, [messages]);

    function clearFocusState() {
        setSelectedAnchorId("");
        setAnchorItems([]);
    }

    function handleFileChange(event) {
        const file = event.target.files && event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const result = e.target && e.target.result ? e.target.result : null;
            setCurrentImageBase64(result);
            clearFocusState();
        };
        reader.readAsDataURL(file);
    }

    function handleRemoveImage() {
        setCurrentImageBase64(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }

    async function handleResetSession() {
        const sid = sessionIdRef.current;
        try {
            if (api.resetSession) {
                await api.resetSession(sid);
            } else {
                await fetch("/api/session/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sid }),
                });
            }
        } catch (_) {
            // Ignore reset errors and clear UI anyway.
        }

        setMessages([]);
        setChatInput("");
        setCurrentImageBase64(null);
        setLatestMeta(null);
        setModalItem(null);
        clearFocusState();
        setQuickActions((bootstrap.suggested_prompts || DEFAULT_BOOTSTRAP.suggested_prompts).slice(0, 6));
    }

    async function sendMessage() {
        const payloadText = chatInput.trim();
        const payloadImage = currentImageBase64;
        if (!payloadText && !payloadImage) return;

        setMessages((prev) => prev.concat([{ id: uid(), role: "user", text: payloadText, image: payloadImage }]));
        setChatInput("");
        handleRemoveImage();
        setIsLoading(true);

        try {
            const requestPayload = {
                text: payloadText,
                image: payloadImage,
                session_id: sessionIdRef.current,
                selected_anchor_id: selectedAnchorId || null,
                new_image_context: Boolean(payloadImage),
                response_mode: "rich",
                include_debug: true,
                max_ui_items: 10,
            };

            const data = api.chat
                ? await api.chat(requestPayload)
                : await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestPayload),
                }).then((response) => response.json());

            const normalizedItems = normalizeItems(data.items);
            const ui = data.ui || {};
            const cards = normalizeItems(ui.cards || normalizedItems);
            const focusOptions = normalizeItems(ui.anchor_options || normalizedItems);
            const actions = Array.isArray(ui.quick_actions) && ui.quick_actions.length > 0
                ? ui.quick_actions.slice(0, 6)
                : quickActions;

            setMessages((prev) => prev.concat([
                {
                    id: uid(),
                    role: "ai",
                    text: data.answer || "",
                    intent: data.intent || "",
                    intentInfo: data.intent_info || null,
                    items: normalizedItems,
                    cards: cards,
                    meta: data.meta || null,
                    error: false,
                },
            ]));

            if (focusOptions.length > 0) {
                setAnchorItems(focusOptions.slice(0, 10));
                const preferredFocus = formatArticleId(data.anchor_article_id)
                    || formatArticleId(focusOptions[0].article_id)
                    || "";
                setSelectedAnchorId(preferredFocus);
            }

            setLatestMeta(data.meta || null);
            setQuickActions(actions);
        } catch (_) {
            setMessages((prev) => prev.concat([
                {
                    id: uid(),
                    role: "ai",
                    text: "Cannot connect to the AI server.",
                    intent: "",
                    intentInfo: null,
                    items: [],
                    cards: [],
                    meta: null,
                    error: true,
                },
            ]));
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <>
            {!started && <HeroScreen bootstrap={bootstrap} onStart={() => setStarted(true)} />}

            {started && (
                <Workspace
                    sessionId={sessionIdRef.current}
                    messages={messages}
                    quickActions={quickActions}
                    selectedAnchorId={selectedAnchorId}
                    anchorItems={anchorItems}
                    onSelectAnchor={(aid) => setSelectedAnchorId(formatArticleId(aid))}
                    onOpenItem={(item) => setModalItem(item)}
                    onPickQuickAction={(text) => {
                        setChatInput(text);
                        if (textAreaRef.current) textAreaRef.current.focus();
                    }}
                    onResetSession={handleResetSession}
                    onSend={sendMessage}
                    chatInput={chatInput}
                    setChatInput={setChatInput}
                    currentImageBase64={currentImageBase64}
                    onUploadClick={() => fileInputRef.current && fileInputRef.current.click()}
                    onFileChange={handleFileChange}
                    onRemoveImage={handleRemoveImage}
                    fileInputRef={fileInputRef}
                    textAreaRef={textAreaRef}
                    isLoading={isLoading}
                    chatBoxRef={chatBoxRef}
                    latestMeta={latestMeta}
                />
            )}

            <ProductModal item={modalItem} onClose={() => setModalItem(null)} />
        </>
    );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
